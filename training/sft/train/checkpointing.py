"""Retention for the numbered checkpoint directories produced by SFT."""
from pathlib import Path
import re
import shutil


def validate_checkpoint_limit(limit):
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("experiment.checkpoints_total_limit must be a positive integer or null")


def prune_checkpoints(output_dir, limit, current_name):
    """Remove oldest numbered checkpoints, protecting the just-saved directory.

    Only direct, non-linked ckpt_<integer> directories are eligible. Call this
    on rank zero after the new model, tokenizer and metadata have been saved.
    """
    validate_checkpoint_limit(limit)
    if limit is None:
        return []
    root = Path(output_dir).resolve()
    checkpoints = []
    for path in root.iterdir():
        match = re.fullmatch(r"ckpt_([0-9]+)", path.name)
        if not match or path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            continue
        if path.is_dir() and path.resolve().parent == root:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda item: (item[0], item[1].name))
    excess = max(0, len(checkpoints) - limit)
    candidates = [path for _, path in checkpoints if path.name != current_name]
    removed = []
    for path in candidates[:excess]:
        # Recheck containment immediately before recursive deletion.
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)() or path.resolve().parent != root:
            raise ValueError(f"Refusing to remove checkpoint outside output directory: {path}")
        shutil.rmtree(path)
        removed.append(path.name)
    return removed
