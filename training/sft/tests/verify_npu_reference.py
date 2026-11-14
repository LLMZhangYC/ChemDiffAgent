"""Compare a tiny one-layer NPU model against the CPU FP32 reference.

This is NOT a CUDA FlashAttention comparison or an 8B quality benchmark.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch
import torch_npu
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import SDARForCausalLM


def difference(actual, expected):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    delta = actual - expected
    return {"max_abs": delta.abs().max().item(),
            "relative_l2": (delta.norm() / expected.norm().clamp_min(1e-12)).item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cpu = SDARForCausalLM.from_pretrained(args.checkpoint).eval()
    if len(cpu.model.layers) != 1 or cpu.config.hidden_size > 256:
        raise ValueError("Use a tiny one-layer fixture, not the public full-size checkpoint")
    # Same BF16-rounded weights on both sides; isolate execution precision.
    torch.npu.set_device(0)
    with torch.no_grad():
        for parameter in cpu.parameters():
            parameter.data = parameter.to(torch.bfloat16).float()
    npu = copy.deepcopy(cpu).to(device="npu:0").eval()
    with torch.no_grad():
        for parameter in npu.parameters():
            parameter.data = parameter.to(torch.bfloat16)
    prompt, response, block = 8, 12, 4
    n = prompt + 2 * response
    idx = torch.arange(n)
    is_prompt, noisy = idx < prompt, idx >= prompt + response
    relative = (idx - prompt).remainder(response) // block
    row, col = relative[:, None], relative[None, :]
    prompt_block = (idx - prompt) // block
    prompt_view = is_prompt[None, :] & (prompt_block[:, None] >= prompt_block[None, :])
    clean_view = ~noisy[None, :] & torch.where(noisy[:, None], row > col, row >= col)
    noisy_view = noisy[:, None] & noisy[None, :] & (row == col)
    attention = torch.where(is_prompt[:, None], prompt_view, is_prompt[None, :] | clean_view | noisy_view)[None, None]
    positions = torch.cat([torch.arange(prompt + response), torch.arange(prompt, prompt + response)])[None]
    ids = (torch.arange(prompt + response) % (cpu.config.vocab_size - 1) + 1)[None]
    # A fixed excluded tool span and identical noise positions on both devices.
    eligible = torch.ones(response, dtype=torch.bool)
    eligible[4:8] = False
    noise = eligible & (torch.arange(response) % 2 == 0)
    extended = torch.cat([ids, ids[:, prompt:].masked_fill(noise, 2)], dim=1)

    def run(model, device):
        logits = model(input_ids=extended.to(device), attention_mask=attention.to(device),
                       position_ids=positions.to(device), use_cache=False).logits
        tail = F.log_softmax(logits[:, -response:].float(), -1)
        logp = tail.gather(-1, ids[:, prompt:].to(device).unsqueeze(-1)).squeeze(-1)
        loss = -(logp * noise.to(device)).sum() / eligible.sum()
        loss.backward()
        grads = []
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError(f"Invalid gradient: {name}")
            grads.append(parameter.grad.detach().float().cpu().flatten())
        return logits, loss, torch.cat(grads)

    expected, cpu_loss, cpu_grad = run(cpu, "cpu")
    actual, npu_loss, npu_grad = run(npu, "npu:0")
    report = {"reference": "CPU FP32 with identical BF16-rounded tiny weights",
              "logits": difference(actual, expected), "gradients": difference(npu_grad, cpu_grad),
              "cpu_loss": cpu_loss.item(), "npu_loss": npu_loss.item()}
    # BF16 compute tolerance: relative vector error < 5%, loss abs error < .02.
    assert report["logits"]["relative_l2"] < 0.05, report
    assert report["gradients"]["relative_l2"] < 0.05, report
    assert abs(cpu_loss.item() - npu_loss.item()) < 0.02, report

    # A single-token cached read must respect an explicit nontrivial mask.
    for model in (cpu, npu):
        model.zero_grad(set_to_none=True)
    all_ids = torch.tensor([[5, 6, 7, 8, 5, 6, 7]])
    full_mask = torch.ones(1, 1, 7, 7, dtype=torch.bool)
    full_mask[:, :, :6, 6] = False
    full_mask[:, :, 6, [1, 3]] = False
    with torch.no_grad():
        full = cpu(input_ids=all_ids, attention_mask=full_mask, use_cache=False).logits[:, -1]
        prefill = npu(input_ids=all_ids[:, :6].to("npu"),
                      attention_mask=full_mask[:, :, :6, :6].to("npu"), use_cache=True, store_kv=True)
        cache = prefill.past_key_values
        decoded = npu(input_ids=all_ids[:, 6:].to("npu"),
                      attention_mask=full_mask[:, :, 6:, :].to("npu"),
                      position_ids=torch.tensor([[6]], device="npu"),
                      past_key_values=cache, use_cache=True, store_kv=False).logits[:, 0]
        assert cache.get_seq_length() == 6, "A read-only cache call unexpectedly appended tokens"
    report["masked_cache_decode"] = difference(decoded, full)
    assert report["masked_cache_decode"]["relative_l2"] < 0.05, report
    report["status"] = "PASS"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
