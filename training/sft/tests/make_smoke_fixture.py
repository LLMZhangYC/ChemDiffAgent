"""Create clearly labelled RANDOM tiny checkpoints/configs for integration tests."""
import argparse
import json
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_reduction_and_truncation import SDARForCausalLM, tiny_config, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(91)
    model = SDARForCausalLM(tiny_config())
    model.save_pretrained(output / "random-3l", safe_serialization=True, max_shard_size="30KB")
    tokenizer().save_pretrained(output / "random-3l")
    for name, response_length, accumulation in [("short", 64, 1), ("8k", 6000, 1), ("accum", 64, 2)]:
        data = [{"prompt": "p " * 128,
                 "response": "r " * 8 + "<tool_response> tool result </tool_response> " + "r " * response_length}] * accumulation
        data_path = output / f"data-{name}.json"
        data_path.write_text(json.dumps(data))
        base = Path(__file__).resolve().parents[1] / "configs/chemdiffagent_sft_npu_1l.yaml"
        config = OmegaConf.load(base)
        config.model.reduction.source_model = str(output / "random-3l")
        config.model.pretrained_model = str(output / "random-1l")
        config.dataset.path = str(data_path)
        config.experiment.output_dir = str(output / f"trained-{name}")
        config.runtime.work_dir = str(output / "runtime")
        config.training.num_train_epochs = 1
        config.training.gradient_accumulation_steps = accumulation
        config.optimizer.params.learning_rate = 1.0e-3
        config.wandb.enabled = False
        OmegaConf.save(config, output / f"smoke-{name}.yaml")
    print(output)


if __name__ == "__main__":
    main()
