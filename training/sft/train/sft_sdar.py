import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union

from omegaconf import OmegaConf
import torch
from torch.optim import AdamW

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed


from models import SDARForCausalLM
from transformers import AutoTokenizer
from train.prompting_utils import UniversalPrompting
from train.checkpointing import prune_checkpoints, validate_checkpoint_limit
from train.validation import split_training_data, evaluate
from train.profiling import create_profiler
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import Dataset, DataLoader

SYSTEM_PROMPT_LEN = 28

from train.utils import get_config
from train.metrics import MetricsLogger
from train.timing import MicrostepTimer
from train.diagnostics import PhaseTrace, configure_deepspeed_overlap, verify_deepspeed_overlap

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")

class TrainDataset(Dataset):
    def __init__(self, extended_input_ids, p_mask, tok_idx_ext, labels):
        self.extended_input_ids = extended_input_ids
        self.p_mask = p_mask
        self.tok_idx_ext = tok_idx_ext
        self.labels = labels

    def __len__(self):
        return len(self.extended_input_ids)

    def __getitem__(self, idx):
        return (
            self.extended_input_ids[idx],
            self.p_mask[idx],
            self.tok_idx_ext[idx],
            self.labels[idx]
        )

def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    project_name = config.experiment.project
    output_dir = config.experiment.get("output_dir", project_name)
    pretrained_model = config.model.pretrained_model
    requested_device = config.training.get("device", "auto")
    if requested_device == "npu":
        import torch_npu  # Register NPU before Accelerator/device initialization.
        if not torch.npu.is_available():
            raise RuntimeError("NPU requested but unavailable; check CANN and torch_npu")

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32 and requested_device != "npu" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        project_dir=config.experiment.logging_dir,
        split_batches=False,
        # This script already defines scheduler length in global optimizer updates.
        # Avoid AcceleratedScheduler multiplying each step by the world size.
        step_scheduler_with_optimizer=False,
    )
    if requested_device == "npu" and accelerator.device.type != "npu":
        raise RuntimeError(f"Expected NPU, got {accelerator.device}")
    if requested_device == "npu" and accelerator.distributed_type != DistributedType.DEEPSPEED:
        raise RuntimeError("Use tools/launch_npu.py: this SFT loop relies on DeepSpeed gradient accumulation")
    overlap_comm = config.training.get("deepspeed_overlap_comm")
    configure_deepspeed_overlap(accelerator, overlap_comm)

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        config_path = Path(output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    metrics_logger = MetricsLogger(output_dir, enabled=accelerator.is_main_process)
    def synchronize_device():
        if accelerator.device.type != "cpu":
            torch.get_device_module(accelerator.device).synchronize(accelerator.device)
    trace = PhaseTrace(config.get("diagnostics", {}), output_dir,
                       accelerator.process_index, synchronize_device)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")


    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    uni_prompting = UniversalPrompting(tokenizer, max_prompt_len=config.training.max_prompt_len,
                                       max_gen_length=config.training.max_gen_length,
                                       ignore_id=-100,
                                       max_seq_length=config.training.get("max_seq_length"),
                                       mask_tool_responses=config.training.get("mask_tool_responses", True))

    #from transformers import AutoModelForCausalLM
    #model = AutoModelForCausalLM.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype="auto")
    # Check depth from metadata BEFORE constructing a model. A full checkpoint
    # accidentally passed here must never allocate all its layers first.
    from models.sdar.configuration_sdar import SDARConfig
    model_config = SDARConfig.from_pretrained(pretrained_model)
    expected_depth = config.model.get("reduction", {}).get("num_hidden_layers")
    if expected_depth is not None and model_config.num_hidden_layers != expected_depth:
        raise ValueError("Checkpoint depth does not match model.reduction.num_hidden_layers; run the export action first")
    with trace.phase("model_load"):
        model, loading_info = SDARForCausalLM.from_pretrained(
            pretrained_model, config=model_config, torch_dtype="auto", output_loading_info=True)
    issues = {key: value for key, value in loading_info.items() if value}
    if issues or len(model.model.layers) != model.config.num_hidden_layers:
        raise RuntimeError(f"Checkpoint did not load completely: {issues}")
    if requested_device == "npu":
        # Avoid transferring file-mapped safetensors pages directly to the NPU.
        # ZeRO-3 already manages partitioned parameter storage during loading.
        if not any(hasattr(p, "ds_id") for p in model.parameters()):
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.data = parameter.detach().to(device="cpu", copy=True)
    logger.info(f"Loaded SDAR with {len(model.model.layers)} decoder layer(s)")

    # calculate loss ourselves, needs logits，so aviod fuse CE
    if hasattr(model, "config"):
        model.config.fuse_cross_entropy = False  
        model.config.use_cache = False
    

    if config.training.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "config"):
            model.config.use_cache = False
    else:
        model = model.to(accelerator.device)

    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # SDAR without projection biases can have an empty no-decay group.
    # DeepSpeed removes empty groups; create the scheduler against the same
    # nonempty groups to avoid a param-group mismatch on newer PyTorch.
    optimizer_grouped_parameters = [group for group in optimizer_grouped_parameters if group["params"]]
    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    def collapse_k_unique(lst, k: int):
        if k <= 0:
            raise ValueError("k must be > 0")
        uniq = sorted(set(lst))

        mapping = {}
        n = len(uniq)
        for idx, val in enumerate(uniq):
            group = idx // k
            end_idx = min((group + 1) * k - 1, n - 1)
            rep = uniq[end_idx]
            mapping[val] = rep
        return [mapping[x] for x in lst]
    
    


    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")


    def simple_collate(batch):
        extended_input_ids, p_mask, tok_idx_ext, labels = zip(*batch)     # Tensor(L)
        return {
            "extended_input_ids":  torch.stack(extended_input_ids),
            "p_mask":  torch.stack(p_mask),
            "tok_idx_ext":  torch.stack(tok_idx_ext),
            "labels":  torch.stack(labels)
        }
    
    dataset_path = config.dataset.get("path", None)
    if dataset_path is None:
        dataset_path = "../data/" + config.dataset.optimization_data + ".json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_load = json.load(f)
    max_samples = config.dataset.get("max_samples", None)
    if max_samples is not None:
        dataset_load = dataset_load[:int(max_samples)]

    split_seed = config.dataset.get("split_seed", config.training.seed if config.training.seed is not None else 10086)
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise ValueError("dataset.split_seed must be an integer")
    validation_seed = config.dataset.get("validation_seed", split_seed + 1)
    if isinstance(validation_seed, bool) or not isinstance(validation_seed, int):
        raise ValueError("dataset.validation_seed must be an integer")
    validation_config = config.get("validation", {})
    eval_steps = validation_config.get("eval_steps")
    if eval_steps is not None and (isinstance(eval_steps, bool) or not isinstance(eval_steps, int) or eval_steps < 1):
        raise ValueError("validation.eval_steps must be a positive integer or null")
    train_rows, validation_rows, split_manifest = split_training_data(
        dataset_load, config.dataset.get("validation_fraction", 0.0), split_seed)
    train_source_count = len(train_rows)
    validation_enabled = bool(validation_rows)
    dataset_load = train_rows + validation_rows
    if accelerator.is_main_process:
        (Path(output_dir) / "data_split.json").write_text(
            json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")

    prompt_list = []
    response_list = []
    step_map_list = []
    for x in dataset_load:
        prompt_list.append(x["prompt"])
        response_list.append(x["response"])
    input_ids_lm, _, start_pos, drop_num = uni_prompting((prompt_list, response_list))
    validation_sources = torch.tensor(
        [i >= train_source_count for i in uni_prompting.kept_indices], dtype=torch.bool)
    dataset_load = [dataset_load[i] for i in uni_prompting.kept_indices]
    if uni_prompting.stats:
        logger.info(f"Sequence preprocessing: {uni_prompting.stats}")
        if accelerator.is_main_process:
            (Path(output_dir) / "preprocessing_report.json").write_text(
                json.dumps(uni_prompting.stats, indent=2) + "\n")
    _, L = input_ids_lm.shape
    L0    = start_pos
    L1    = L - L0
    post_num = config.training.post_num
    max_seq_length = config.training.get("max_seq_length")
    if max_seq_length is not None and L0 + 2 * L1 > max_seq_length:
        raise RuntimeError("Expanded sequence exceeds training.max_seq_length")

    for x in dataset_load:
        if "step_map" not in x.keys():
            step_map_list.append([j for j in range(L1)])
        else:
            step_map_i = x["step_map"]
            if len(step_map_i) > L1:
                step_map_i = step_map_i[:L1]
            else:
                step_map_i = step_map_i + [max(step_map_i, default=-1) + 1] * (L1 - len(step_map_i))
            step_map_list.append(step_map_i)
    
    def make_basic_block_attention(
        N: int,
        start_pos: int,            # = L0
        block_size: int,           # = b
    ) -> torch.Tensor:
        B = 1
        L0     = start_pos
        L1     = (N - L0) // 2          # N = L0 + 2·L1 
        assert L0 + 2 * L1 == N, "input length must be L0 + 2*L1"

        # all -inf first
        bias = torch.zeros((B, 1, N, N), dtype=torch.bool)


        rows = torch.arange(L0 + L1, L0 + 2 * L1)              # (L1,)
        rows_token = torch.arange(L0, L0 + L1)              # (L1,)

        # update block by block
        for bi in range((L1 + block_size - 1) // block_size):
            #  [bi*b , min((bi+1)*b, L1))
            left_end   = L0 + min((bi) * block_size, L1)        
            right_start= L0 + L1 + (left_end - L0)

            i_start = bi * block_size
            i_end   = min((bi + 1) * block_size, L1)              # no i_end

            block_rows = rows[i_start:i_end]                    
            bias[:, :, block_rows.unsqueeze(-1), 0:left_end]   = 1
            bias[:, :, block_rows.unsqueeze(-1), right_start:(right_start + block_size)] = 1

            block_rows = rows_token[i_start:i_end]
            left_end   = L0 + min((bi + 1) * block_size, L1)
            bias[:, :, block_rows.unsqueeze(-1), 0:left_end]   = 1
        
        if L0 > 0:
            num_blocks_pre = (L0 + block_size - 1) // block_size
            for bi in range(num_blocks_pre):
                # row interval [row_start, row_end)
                row_end   = max(L0 - bi * block_size, 0)
                row_start = max(L0 - (bi + 1) * block_size, 0)
                if row_end > row_start:
                    block_rows = torch.arange(row_start, row_end)
                    bias[:, :, block_rows.unsqueeze(-1), 0:row_end] = 1
        
        return bias        # (B,1,N,N)
    
    
    basic_block_attention = make_basic_block_attention(L0 + 2 * L1, start_pos, config.training.block_size)
    basic_block_attention = basic_block_attention.cpu()


    def process_pad(attn, input_ids):
        N = L0 + 2 * L1
        device = input_ids.device

        cols = torch.arange(N, device=device)                  # (N,)
        key_mask = (cols < start_pos).unsqueeze(0) & (input_ids == pad_id)  # (B, N)

        # set -inf
        attn.masked_fill_(key_mask[:, None, None, :], 0)

        # aviod +-inf or none in forward
        A = attn[:, 0]  # (B, N, N)
        bad = (A.sum(dim=-1) == 0) & (torch.arange(A.size(1), device=A.device).unsqueeze(0) < start_pos)
        b, r = bad.nonzero(as_tuple=True)
        A[b, r, :] = 0; A[b, r, r] = 1  

        return attn





    def one_round_vectorized(input_ids_b, step_map_b, L0, L1, block_size, mask_id):
        """
        Perform a single "round" on one sample b:
        - For each block, take the minimum non -1 value in step_map.
        - Create pmask (positions equal to the block minimum).
        - Create a noise mask for the extended segment (positions >= block minimum).
        - Mark the chosen minimum positions in step_map as -1 for the next round.

        Returns:
        extended_input_ids_b : Tensor with duplicated + masked response segment
        pmask_b              : Boolean mask for tokens selected in this round
        new_step_map_b       : Updated step_map (selected positions set to -1)
        has_any              : Whether any position was selected in this round
        """
        device = input_ids_b.device
        NB = (L1 + block_size - 1) // block_size
        pad_len = NB * block_size - L1

        # Reshape step_map into [NB, block_size], fill last incomplete block with -1
        step_pad = torch.full((NB * block_size,), -1, dtype=torch.long, device=device)
        step_pad[:L1] = step_map_b
        step_blk = step_pad.view(NB, block_size)                      # [NB, Bk]

        valid = step_blk.ge(0)                                        # Valid positions (not -1)
        big = torch.iinfo(step_blk.dtype).max
        tmp = step_blk.masked_fill(~valid, big)                       # Fill invalid positions with a large value
        min_vals, _ = tmp.min(dim=1, keepdim=True)                    # Current minimum for each block

        # Select positions equal to block minimum (only valid positions)
        pmask_blk = step_blk.eq(min_vals) & valid                     
        if not pmask_blk.any():
            # No positions left to select in this round
            return None, None, step_map_b, False

        # Noise mask for extended segment: mark positions >= block minimum
        ge_mask_blk = step_blk.ge(min_vals) & valid                   # [NB, Bk]

        # Flatten back to length L1 (discard padding)
        pmask_tail = pmask_blk.view(-1)[:L1]                          # [L1]
        ge_mask_tail = ge_mask_blk.view(-1)[:L1]                      # [L1]

        # Construct pmask_b: [0:L0] = False, [L0:] = pmask_tail
        pmask_b = torch.zeros(L0 + L1, dtype=torch.bool, device=device)
        pmask_b[L0:] = pmask_tail

        # Build extended segment: duplicate response and replace noise positions with mask_id
        tail = input_ids_b[L0:L0+L1].clone()
        tail[ge_mask_tail] = mask_id

        
        extended_input_ids_b = torch.empty(L0 + L1 + L1, dtype=input_ids_b.dtype, device=device)
        extended_input_ids_b[:L0+L1] = input_ids_b
        extended_input_ids_b[L0+L1:] = tail

        # Update step_map: mark selected minimum positions as -1 for the next round
        new_step_map_b = step_map_b.clone()
        new_step_map_b[pmask_tail] = -1

        return extended_input_ids_b, pmask_b, new_step_map_b, True



    def find_token_positions(input_ids: torch.Tensor, token_id: int, start_pos: int) -> list[torch.Tensor]:

        if input_ids.dim() != 2:
            raise ValueError("input_ids must be a 2D tensor (batch_size, sequence_length)")
        
        batch_size = input_ids.shape[0]
        
        search_area = input_ids[:, start_pos:]
        
        match_mask = (search_area == token_id)  
        
        results = []
        for batch_idx in range(batch_size):
            relative_positions = torch.nonzero(match_mask[batch_idx], as_tuple=False).squeeze(-1)
            
            absolute_positions = relative_positions + start_pos
            
            results.append(absolute_positions)
        
        return results
    def collect_training_data(input_ids: torch.tensor, step_map_list, generator=None):

        B, L = input_ids.shape
        L0    = start_pos
        L1    = L - L0
        # print(tokenizer('\n</tool_response><|im_end|>'))
        mask_tool_responses = config.training.get("mask_tool_responses", True)
        if mask_tool_responses:
            L2_s = find_token_positions(input_ids, tokenizer.convert_tokens_to_ids('<tool_response>'), start_pos)
            L2_e = find_token_positions(input_ids, tokenizer.convert_tokens_to_ids('</tool_response>'), start_pos)

        block_size = config.training.block_size

        lower = config.training.lower_p
        upper = config.training.upper_p


        
        if config.training.method == "semi-ar":

            extended_input_ids_list, pmask_list = [], []

            # Pre-compute the global linear probability
            
            for b in range(B):

                prob_ramp = torch.empty(L1).uniform_(lower, upper, generator=generator)
                rand_tail = torch.rand(L1, generator=generator)
                pmask_tail = rand_tail <= prob_ramp  # [L1], True means to mask

                if mask_tool_responses:
                    # Baseline behavior: tool results are context and never prediction targets.
                    assert len(L2_s[b]) == len(L2_e[b]), "unbalanced tool response tags"
                    for begin, end in zip(L2_s[b], L2_e[b]):
                        pmask_tail[begin - start_pos:end - start_pos + 1] = False
                # 2) Construct the pmask for this sample (prefix all False, tail uses pmask_tail)
                pmask_b = torch.cat([
                    torch.zeros(L0, dtype=torch.bool),
                    pmask_tail
                ], dim=0)  # [L]

                # 3) Construct the noisy tail and append it to the original sequence to get the extended sequence
                noise_tail = input_ids[b, L0:].clone()            # [L1]
                noise_tail.masked_fill_(pmask_tail, mask_id)      # Replace masked positions
                extended_b = torch.cat([input_ids[b], noise_tail], dim=0)  # [L + L1]

                extended_input_ids_list.append(extended_b)
                pmask_list.append(pmask_b)

            
        elif config.training.method == "trace":

            for b in range(B):
                step_map_i = step_map_list[b]

                for j in range(int((L1 - 1) / block_size) + 1):
                    start = j * block_size
                    end = min(L1, (j + 1) * block_size)
                    step_map_list[b][start:end] = collapse_k_unique(step_map_i[start:end], config.training.shrink)
            
            step_map = torch.as_tensor(step_map_list, dtype=torch.long)

            assert step_map.shape[1] == L1

            extended_input_ids_list, pmask_list = [], []

            for b in range(B):
                step_b = step_map[b]
                while True:
                    out = one_round_vectorized(
                        input_ids_b=input_ids[b],
                        step_map_b=step_b,
                        L0=L0,
                        L1=L1,
                        block_size=block_size,
                        mask_id=mask_id,
                    )
                    extended_b, pmask_b, step_b, has_any = out
                    if not has_any:
                        break

                    extended_input_ids_list.append(extended_b)
                    pmask_list.append(pmask_b)

        extended_input_ids = torch.stack(extended_input_ids_list, dim=0)
        p_mask =  torch.stack(pmask_list, dim=0).to(torch.bool)
        
        pad_resp = (extended_input_ids[:, :L] == pad_id) & p_mask        
        if post_num is not None:
            cum_pad = torch.cumsum(pad_resp.int(), dim=1)
            p_mask &= ~(pad_resp & (cum_pad > post_num))
        
        labels = extended_input_ids[:, :L].clone()

        idx = torch.arange(L).unsqueeze(0).expand(extended_input_ids.shape[0], -1)
        valid = (idx >= start_pos) | extended_input_ids[:, :L].ne(pad_id)      
        tok_idx = valid.long().cumsum(dim=-1) - 1         
        tok_idx = tok_idx.masked_fill(~valid, 1)
        tok_idx_resp = tok_idx[:, start_pos:]  
        tok_idx_ext  = torch.cat([tok_idx, tok_idx_resp], dim=1)

        keep = p_mask.view(p_mask.size(0), -1).any(dim=1)

        extended_input_ids = extended_input_ids[keep]
        p_mask            = p_mask[keep]
        tok_idx_ext       = tok_idx_ext[keep]
        labels            = labels[keep]

        return extended_input_ids, p_mask, tok_idx_ext, labels
        

    
    if not (~validation_sources).any():
        raise ValueError("No training samples remain after prompt/response preprocessing")
    if validation_enabled and not validation_sources.any():
        raise ValueError("No validation samples remain after prompt/response preprocessing")
    train_step_maps = [row for row, is_val in zip(step_map_list, validation_sources.tolist()) if not is_val]
    extended_input_ids, p_mask, tok_idx_ext, labels = collect_training_data(
        input_ids_lm[~validation_sources], train_step_maps)
    if not len(extended_input_ids):
        raise ValueError("No supervised tokens survived noise sampling; check data and masking probabilities")
    if max_seq_length is not None and extended_input_ids.shape[1] > max_seq_length:
        raise RuntimeError("Actual model input exceeds training.max_seq_length")




    dataset_lm = TrainDataset(extended_input_ids, p_mask, tok_idx_ext, labels)
    validation_dataloader = None
    validation_dataset = None
    if validation_enabled:
        val_step_maps = [row for row, is_val in zip(step_map_list, validation_sources.tolist()) if is_val]
        # A separate CPU generator keeps held-out corruption fixed without changing training RNG.
        validation_tensors = collect_training_data(
            input_ids_lm[validation_sources], val_step_maps,
            generator=torch.Generator().manual_seed(validation_seed))
        validation_dataset = TrainDataset(*validation_tensors)
        if not len(validation_dataset):
            raise ValueError("No validation masked targets survived corruption; change validation_seed/data")
        validation_dataloader = DataLoader(
            validation_dataset, batch_size=config.training.batch_size_lm, shuffle=False,
            collate_fn=simple_collate, num_workers=0,
            generator=torch.Generator().manual_seed(validation_seed))
    split_manifest.update(train_preprocessed_samples=int((~validation_sources).sum()),
                          validation_preprocessed_samples=int(validation_sources.sum()),
                          train_examples=len(dataset_lm),
                          validation_examples=len(validation_dataset) if validation_enabled else 0,
                          validation_seed=validation_seed)
    split_manifest["dataset_path"] = str(dataset_path)
    if accelerator.is_main_process:
        (Path(output_dir) / "data_split.json").write_text(
            json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")

    total_batch_size_lm = config.training.batch_size_lm * accelerator.num_processes * config.training.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(dataset_lm) / total_batch_size_lm)
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch * num_train_epochs + 1
    # Optional cap on actual optimizer updates, not forward/backward micro-steps.
    max_optimizer_steps = config.training.get("max_steps")
    if max_optimizer_steps is not None:
        if isinstance(max_optimizer_steps, bool) or not isinstance(max_optimizer_steps, int) or max_optimizer_steps < 1:
            raise ValueError("training.max_steps must be a positive integer or null")
        max_train_steps = min(max_train_steps, max_optimizer_steps)

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale
    )

    train_dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        shuffle=True,
        collate_fn=simple_collate,
        num_workers=0
    )

    ##################################
    #       Prepare accelerator     #
    #################################
    logger.info("Preparing model, optimizer and dataloaders")
    with trace.phase("accelerator_prepare"):
        if validation_enabled:
            model, optimizer, lr_scheduler, train_dataloader_lm, validation_dataloader = accelerator.prepare(
                model, optimizer, lr_scheduler, train_dataloader_lm, validation_dataloader)
        else:
            model, optimizer, lr_scheduler, train_dataloader_lm = accelerator.prepare(
                model, optimizer, lr_scheduler, train_dataloader_lm)
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        actual_overlap = verify_deepspeed_overlap(model, overlap_comm)
        logger.info(f"DeepSpeed effective overlap_comm={actual_overlap}; "
                    f"gradient_accumulation_steps={model.gradient_accumulation_steps()}")
        trace.wrap_deepspeed(model)

    #################################
    #             Training          #
    #################################
    logger.info("***** Running training *****")
    
    logger.info(f"  Num response = {len(dataset_load)}")
    logger.info(f"  Num sample dropped = {drop_num}")
    logger.info(f"  Num training data = {extended_input_ids.shape[0]}")
    logger.info(f"  Num validation data = {len(validation_dataset) if validation_enabled else 0}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {config.training.batch_size_lm}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    first_epoch = 0

    import torch.nn.functional as F

    def forward_process(extended_input_ids, p_mask, tok_idx_ext, labels, return_metrics=False):

        B, L = p_mask.shape
        L0    = start_pos
        L1    = L - L0
        device = extended_input_ids.device

        with trace.phase("attention_mask"):
            attention_mask = basic_block_attention.clone()
            attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
            attention_mask = process_pad(attention_mask, extended_input_ids)

        with trace.phase("model_forward", input_shape=list(extended_input_ids.shape)):
            logits = model(input_ids = extended_input_ids, attention_mask = attention_mask, position_ids = tok_idx_ext).logits
        with trace.phase("loss_compute"):
            logits = torch.cat([logits[:, :L0, :], logits[:, L0 + L1 :, :]], dim=1)  # (B, L0+L1, V)

            log_probs = F.log_softmax(logits, dim=-1)
            logp_tok  = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)     # (B, T)
            # Normalize by all non-padding response tokens. In standard SFT this includes
            # every token inside <tool_response>...</tool_response>.
            response_token_count = extended_input_ids[:, L0:L].ne(pad_id).sum(dim=1).clamp_min(1)
            if config.training.get("mask_tool_responses", True):
                tool_begin = find_token_positions(extended_input_ids[:, :L], tokenizer.convert_tokens_to_ids('<tool_response>'), start_pos)
                tool_end = find_token_positions(extended_input_ids[:, :L], tokenizer.convert_tokens_to_ids('</tool_response>'), start_pos)
                excluded = torch.tensor(
                    [sum((end - begin + 1).item() for begin, end in zip(tool_begin[b], tool_end[b])) for b in range(B)],
                    device=device,
                )
                response_token_count = (response_token_count - excluded).clamp_min(1)
            loss_lm = -(logp_tok * p_mask).sum(dim=1) / response_token_count

            if return_metrics:
                masked_nll = -(logp_tok.float() * p_mask).sum(dim=1)
                correct = ((logits.argmax(dim=-1) == labels) & p_mask).sum(dim=1)
                return torch.stack((loss_lm.float(), masked_nll, p_mask.sum(dim=1).float(),
                                    correct.float()), dim=1)

            loss_lm = loss_lm.sum() / B
        
        return loss_lm




    from tqdm.auto import tqdm
    global_steps = 0
    completed_optimizer_steps = 0
    last_validation_step = None
    last_validation_epoch = None
    validation_history = []

    def run_validation(epoch):
        nonlocal last_validation_step, last_validation_epoch
        with trace.phase("validation"):
            metrics = evaluate(model, validation_dataloader,
                                lambda **batch: forward_process(**batch, return_metrics=True), accelerator)
        entry = dict(epoch=epoch, micro_step=global_steps, optimizer_step=completed_optimizer_steps, **metrics)
        validation_history.append(entry)
        last_validation_step = completed_optimizer_steps
        last_validation_epoch = epoch
        logger.info(f"Validation: {entry}")
        if accelerator.is_main_process:
            with (Path(output_dir) / "validation_metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
            metrics_logger.log("validation", epoch, global_steps, completed_optimizer_steps, **metrics)

    for epoch in range(first_epoch, num_train_epochs):
        
        model.train()
        
        progress_bar = tqdm(
            train_dataloader_lm,
            desc=f"Epoch {epoch+1}/{num_train_epochs}",
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,         
            leave=True                
        )
        
        

        # Only the first epoch may create a native profiler. Never restart it.
        profile_settings = config.get("profiling", {}) if epoch == first_epoch else {}
        with create_profiler(profile_settings, output_dir,
                             accelerator.process_index, accelerator.device.type, epoch,
                             world_size=accelerator.num_processes) as profiler:
            timer = MicrostepTimer(accelerator, config.training.get("log_microstep_time", True))
            for step, batch in enumerate(trace.batches(timer.batches(progress_bar), epoch, global_steps + 1), start=1):
                profiling_running = getattr(profiler, "running", False)

                # Accelerate 1.15 explicitly passes sync_gradients to DeepSpeed.
                # Enter this context for every micro-batch to advance the boundary.
                with accelerator.accumulate(model):
                    trace.emit("accumulation", "boundary", sync_gradients=accelerator.sync_gradients)
                    with trace.phase("batch_to_device"):
                        extended_input_ids = batch["extended_input_ids"].to(accelerator.device)
                        p_mask = batch["p_mask"].to(accelerator.device)
                        tok_idx_ext = batch["tok_idx_ext"].to(accelerator.device)
                        labels = batch["labels"].to(accelerator.device)

                    with trace.phase("forward_and_loss"):
                        loss_lm = forward_process(
                            extended_input_ids=extended_input_ids,
                            p_mask=p_mask,
                            tok_idx_ext=tok_idx_ext,
                            labels=labels
                        )
                    with trace.phase("loss_check"):
                        if not torch.isfinite(loss_lm.detach()).all():
                            raise FloatingPointError("Non-finite SFT loss")
                    optimizer_step_before = getattr(model, "global_steps", None)
                    with trace.phase("accelerator_backward"):
                        accelerator.backward(loss_lm)
                    if accelerator.sync_gradients and config.training.max_grad_norm is not None:
                        with trace.phase("gradient_clip"):
                            accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                    # DeepSpeed performs the update inside accelerator.backward();
                    # its optimizer wrapper's step/zero_grad methods are no-ops.
                    with trace.phase("optimizer_wrapper_step"):
                        optimizer.step()
                    optimizer_updated = (model.global_steps > optimizer_step_before
                                         if optimizer_step_before is not None else
                                         accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped)
                    if optimizer_updated:
                        with trace.phase("scheduler_step"):
                            lr_scheduler.step()
                        completed_optimizer_steps += 1
                    with trace.phase("optimizer_zero_grad"):
                        optimizer.zero_grad(set_to_none=True)

                with trace.phase("timing_sync_and_gather"):
                    timing_metrics = timer.finish()
                global_steps += 1
                # All ranks participate; only rank zero writes the global mean.
                with trace.phase("loss_reduce"):
                    step_loss = accelerator.reduce(loss_lm.detach().float(), reduction="mean").item()
                metrics_logger.log("train", epoch, global_steps, completed_optimizer_steps,
                                   loss=step_loss, learning_rate=optimizer.param_groups[0]["lr"],
                                   optimizer_updated=optimizer_updated, profiling_running=profiling_running,
                                   sync_gradients=accelerator.sync_gradients,
                                   scheduler_step=int(getattr(getattr(lr_scheduler, "scheduler", lr_scheduler), "last_epoch", -1)),
                                   **timing_metrics)
                postfix = {"loss": f"{loss_lm.item():.4f}"}
                if timing_metrics:
                    postfix["micro_s"] = f"{timing_metrics['microstep_time_s']:.3f}"
                progress_bar.set_postfix(**postfix)
                if optimizer_updated:
                    # Include all accumulated micro-batches, and advance before early stopping.
                    with trace.phase("profiler_step"):
                        profiler.step()
                if trace.should_stop(global_steps):
                    break
                if (validation_enabled and eval_steps is not None and completed_optimizer_steps > 0
                        and completed_optimizer_steps % eval_steps == 0
                        and last_validation_step != completed_optimizer_steps):
                    run_validation(epoch)
                if max_optimizer_steps is not None and completed_optimizer_steps >= max_optimizer_steps:
                    logger.info(f"Reached training.max_steps={max_optimizer_steps}; saving and stopping")
                    break
        if trace.should_stop(global_steps):
            trace.emit("diagnostics", "stopped", completed_micro_steps=global_steps,
                       optimizer_steps=completed_optimizer_steps, checkpoint_saved=False)
            logger.info(f"Diagnostic stop after {global_steps} micro-steps; no validation or checkpoint save")
            with trace.phase("diagnostic_shutdown"):
                accelerator.wait_for_everyone()
                accelerator.end_training()
            return
        with trace.phase("epoch_barrier"):
            accelerator.wait_for_everyone()
        if (validation_enabled and validation_config.get("eval_after_epoch", True)
                and (last_validation_step, last_validation_epoch) != (completed_optimizer_steps, epoch)):
            run_validation(epoch)
        with trace.phase("checkpoint_save"):
            save_checkpoint(model, tokenizer, config, accelerator, config.model.optimized_name, epoch=epoch)
        if max_optimizer_steps is not None and completed_optimizer_steps >= max_optimizer_steps:
            break

    
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        report = {
            "device": str(accelerator.device),
            "distributed_type": str(accelerator.distributed_type),
            "num_hidden_layers": len(unwrapped.model.layers),
            "expanded_length": int(extended_input_ids.shape[1]),
            "metrics_run_id": metrics_logger.run_id,
            "micro_steps": global_steps,
            "max_steps": max_optimizer_steps,
            "optimizer_steps": int(getattr(model, "global_steps", global_steps)),
            "scheduler_steps": int(getattr(getattr(lr_scheduler, "scheduler", lr_scheduler), "last_epoch", -1)),
            "last_loss": float(loss_lm.detach()),
            "validation": validation_history,
        }
        if accelerator.device.type == "npu":
            report["peak_allocated_gib"] = torch.npu.max_memory_allocated() / 2**30
        (Path(output_dir) / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")

    accelerator.end_training()


def save_checkpoint(model, tokenizer, config, accelerator, name, epoch=None):
    from pathlib import Path
    import time, json, shutil, os, glob, importlib, inspect

    output_dir = Path(config.experiment.get("output_dir", config.experiment.project))
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)
    validate_checkpoint_limit(checkpoints_total_limit)
    ckpt_name = f"ckpt_{epoch}" if epoch is not None else "ckpt"
    save_base = output_dir / ckpt_name
    save_base.mkdir(exist_ok=True)

    model_to_save = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        save_dir = save_base / name
        model_to_save.save_pretrained(
            save_dir,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(str(save_dir))

        def _copy_dynamic_modules(dst_dir, model_obj, tok_obj):
            copied = 0
            modules = set()
            for obj in [model_obj, getattr(model_obj, "config", None), tok_obj]:
                if obj is None:
                    continue
                modname = getattr(obj.__class__, "__module__", None)
                if modname:
                    modules.add(modname)

            for modname in modules:
                try:
                    mod = importlib.import_module(modname)
                    src_file = inspect.getsourcefile(mod)  # e.g. .../modeling_sdar.py
                    if not src_file or not os.path.exists(src_file):
                        continue
                    base_dir = os.path.dirname(src_file)

                    for pattern in ("modeling_*.py", "configuration_*.py", "tokenization_*.py", "processing_*.py"):
                        for fn in glob.glob(os.path.join(base_dir, pattern)):
                            dst = os.path.join(dst_dir, os.path.basename(fn))
                            if os.path.exists(dst):
                                continue
                            shutil.copy2(fn, dst)
                            copied += 1
                except Exception as e:
                    logger.warning(f"Skip copying from module {modname}: {e}")

            logger.info(f"Copied {copied} custom module files into {dst_dir}")

        _copy_dynamic_modules(str(save_dir), model_to_save, tokenizer)

        metadata = {
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with (save_base / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model + tokenizer to {save_dir}")
        removed = prune_checkpoints(output_dir, checkpoints_total_limit, save_base.name)
        if removed:
            logger.info(f"Removed old checkpoints: {', '.join(removed)}")
    accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
