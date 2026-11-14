#!/usr/bin/env python3
"""Export the first N SDAR blocks without loading the full source model.

Usage: python tools/export_reduced_sdar.py config=configs/chemdiffagent_sft_npu_1l.yaml
Only config/model metadata and selected safetensors are read. No NPU is needed.
"""
import json
from collections import defaultdict
from pathlib import Path
import re
import shutil
import tempfile

from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file


LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.")
SFT_ROOT = Path(__file__).resolve().parents[1]


def expected_shapes(config, depth):
    hidden = config["hidden_size"]
    vocab = config["vocab_size"]
    heads = config["num_attention_heads"]
    kv_heads = config.get("num_key_value_heads", heads)
    head_dim = config.get("head_dim", hidden // heads)
    intermediate = config["intermediate_size"]
    result = {
        "model.embed_tokens.weight": (vocab, hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (vocab, hidden),
    }
    for layer in range(depth):
        prefix = f"model.layers.{layer}."
        sizes = {"q_proj": (heads * head_dim, hidden),
                 "k_proj": (kv_heads * head_dim, hidden),
                 "v_proj": (kv_heads * head_dim, hidden),
                 "o_proj": (hidden, heads * head_dim)}
        for name, shape in sizes.items():
            result[prefix + f"self_attn.{name}.weight"] = shape
            if config.get("attention_bias", False):
                result[prefix + f"self_attn.{name}.bias"] = (shape[0],)
        for name in ("q_norm", "k_norm"):
            result[prefix + f"self_attn.{name}.weight"] = (head_dim,)
        for name in ("input_layernorm", "post_attention_layernorm"):
            result[prefix + name + ".weight"] = (hidden,)
        for name in ("gate_proj", "up_proj"):
            result[prefix + f"mlp.{name}.weight"] = (intermediate, hidden)
        result[prefix + "mlp.down_proj.weight"] = (hidden, intermediate)
    return result


def read_index(source):
    index = source / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Empty or invalid safetensors weight_map")
        for shard in weight_map.values():
            if not isinstance(shard, str) or Path(shard).name != shard:
                raise ValueError("Shard names must be local filenames")
        return weight_map
    single = source / "model.safetensors"
    if single.is_file():
        with safe_open(str(single), framework="pt", device="cpu") as handle:
            return dict.fromkeys(handle.keys(), single.name)
    raise ValueError("Expected model.safetensors or model.safetensors.index.json; .bin is not supported")


def export_prefix(source, destination, depth):
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if destination.exists() or destination == source or source in destination.parents:
        raise ValueError("Output must be a NEW directory outside the source checkpoint")
    config = json.loads((source / "config.json").read_text())
    original_depth = config["num_hidden_layers"]
    if config.get("model_type") != "sdar" or config.get("use_sliding_window", False):
        raise ValueError("Only dense SDAR checkpoints with full attention are supported")
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= original_depth:
        raise ValueError(f"num_hidden_layers must be an integer in [1, {original_depth}]")

    weight_map = read_index(source)
    layers = {int(match.group(1)) for key in weight_map if (match := LAYER_KEY.match(key))}
    if layers != set(range(original_depth)):
        raise ValueError("Source layer index does not match config.num_hidden_layers")
    shapes = expected_shapes(config, depth)
    required = set(shapes)
    if config.get("tie_word_embeddings", False):
        required.discard("lm_head.weight")
    selected = {key: shard for key, shard in weight_map.items()
                if not (match := LAYER_KEY.match(key)) or int(match.group(1)) < depth}
    missing, unexpected = required - selected.keys(), selected.keys() - shapes.keys()
    if missing or unexpected:
        raise ValueError(f"Incompatible SDAR weights: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    groups = defaultdict(list)
    for key, shard in selected.items():
        groups[shard].append(key)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sdar-export-", dir=destination.parent) as temp:
        staging = Path(temp) / "checkpoint"
        staging.mkdir()
        total_bytes, output_map = 0, {}
        # Unselected shards are never opened. Within each selected shard, only
        # get_tensor(selected_key) materializes data; this is not a full load/filter.
        for i, (shard, keys) in enumerate(sorted(groups.items()), 1):
            filename = f"model-{i:05d}-of-{len(groups):05d}.safetensors"
            with safe_open(str(source / shard), framework="pt", device="cpu") as handle:
                for key in keys:
                    shape = tuple(handle.get_slice(key).get_shape())
                    if shape != shapes[key]:
                        raise ValueError(f"Shape mismatch for {key}: {shape} != {shapes[key]}")
                tensors = {key: handle.get_tensor(key) for key in sorted(keys)}
                save_file(tensors, str(staging / filename), metadata={"format": "pt"})
                total_bytes += sum(t.numel() * t.element_size() for t in tensors.values())
            output_map.update(dict.fromkeys(keys, filename))
            del tensors

        # Keep tokenizer/vocabulary assets. Export the local compatible SDAR
        # implementation so AutoModel reload does not bring back CUDA-only code.
        for item in source.iterdir():
            if item.is_file() and (item.name.startswith(("tokenizer", "tokenization_", "chat_template"))
                                  or item.name in {"vocab.json", "merges.txt", "special_tokens_map.json",
                                                   "added_tokens.json", "generation_config.json"}
                                  or item.suffix == ".model"):
                shutil.copy2(item, staging / item.name)
        for name in ("configuration_sdar.py", "modeling_sdar.py"):
            shutil.copy2(SFT_ROOT / "models" / "sdar" / name, staging / name)
        config["num_hidden_layers"] = depth
        config["architectures"] = ["SDARForCausalLM"]
        config["fuse_cross_entropy"] = False
        config.setdefault("auto_map", {}).update({
            "AutoConfig": "configuration_sdar.SDARConfig",
            "AutoModel": "modeling_sdar.SDARModel",
            "AutoModelForCausalLM": "modeling_sdar.SDARForCausalLM",
        })
        report = {"source": str(source), "original_layers": original_depth,
                  "num_hidden_layers": depth, "selected_layers": list(range(depth)),
                  "weight_tensors": len(output_map), "weight_bytes": total_bytes,
                  "read_shards": sorted(groups), "selected_keys": sorted(output_map)}
        for name, data in {
            "config.json": config,
            "model.safetensors.index.json": {"metadata": {"total_size": total_bytes}, "weight_map": output_map},
            "reduction_report.json": report,
        }.items():
            (staging / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        staging.rename(destination)
    return report


def main():
    cli = OmegaConf.from_cli()
    if not cli.get("config"):
        raise SystemExit("Pass config=/path/to/sft.yaml")
    config = OmegaConf.merge(OmegaConf.load(cli.config), cli)
    source = config.model.reduction.get("source_model")
    if not source:
        raise SystemExit("Set model.reduction.source_model to the LOCAL public checkpoint directory")
    report = export_prefix(source, config.model.pretrained_model, config.model.reduction.num_hidden_layers)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
