"""Reproducible source-level holdout and distributed masked-token evaluation."""
import hashlib
import json
import math
import random


def split_training_data(rows, fraction=0.1, seed=10086):
    """Keep the same prompt's responses together, before trace expansion.

    Return source indices as well as rows so the split can be audited. Grouping
    can make the held-out size slightly larger than the requested fraction.
    """
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction < 1:
        raise ValueError("dataset.validation_fraction must be in [0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("dataset.split_seed must be an integer")
    if not rows:
        raise ValueError("Cannot split an empty SFT dataset")
    groups = {}
    for index, row in enumerate(rows):
        content = json.dumps(row["prompt"], ensure_ascii=False)
        key = hashlib.sha256(content.encode("utf-8")).hexdigest()
        groups.setdefault(key, []).append(index)
    keys = sorted(groups)
    if fraction and len(keys) < 2:
        raise ValueError("Validation needs at least two distinct prompts")
    random.Random(seed).shuffle(keys)
    validation_indices = set()
    if fraction:
        target = max(1, math.ceil(len(rows) * fraction))
        # Always leave one complete group for training.
        for key in keys[:-1]:
            validation_indices.update(groups[key])
            if len(validation_indices) >= target:
                break
    train_indices = [i for i in range(len(rows)) if i not in validation_indices]
    validation_indices = sorted(validation_indices)
    manifest = {"seed": seed, "validation_fraction": fraction,
                "source_samples": len(rows), "distinct_prompts": len(groups), "group_by": "prompt",
                "train_indices": train_indices, "validation_indices": validation_indices}
    return [rows[i] for i in train_indices], [rows[i] for i in validation_indices], manifest


def evaluate(model, dataloader, forward_metrics, accelerator):
    """Evaluate without updates; gather_for_metrics removes padded final samples.

    forward_metrics returns [B, 4]: response-normalized loss, masked NLL sum,
    masked-token count, correct masked-token count. The primary loss matches
    training's per-response normalization; masked CE is reported separately.
    All ranks must enter this function (including with ZeRO-3).
    """
    import torch

    was_training = model.training
    totals = torch.zeros(4, dtype=torch.float64)
    samples = 0
    try:
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = {name: tensor.to(accelerator.device) for name, tensor in batch.items()}
                metrics = forward_metrics(**batch)
                metrics = accelerator.gather_for_metrics(metrics)
                metrics = metrics.detach().to(device="cpu", dtype=torch.float64)
                if metrics.ndim != 2 or metrics.shape[1] != 4 or not torch.isfinite(metrics).all():
                    raise ValueError("Invalid/non-finite validation metrics")
                totals += metrics.sum(dim=0)
                samples += len(metrics)
    finally:
        model.train(was_training)
    if not samples or totals[2] <= 0:
        raise ValueError("Validation has no supervised masked tokens after preprocessing")
    return {"loss": (totals[0] / samples).item(),
            "masked_cross_entropy": (totals[1] / totals[2]).item(),
            "masked_accuracy": (totals[3] / totals[2]).item(),
            "samples": samples, "masked_tokens": int(totals[2].item())}
