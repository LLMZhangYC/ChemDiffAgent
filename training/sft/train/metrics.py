"""Local, line-delimited metrics for plotting without an external service."""
import json
from pathlib import Path
import uuid


class MetricsLogger:
    def __init__(self, output_dir, enabled=True):
        self.path = Path(output_dir) / "metrics.jsonl"
        self.enabled = enabled
        self.run_id = uuid.uuid4().hex

    def log(self, phase, epoch, micro_step, optimizer_step, **metrics):
        if not self.enabled:
            return
        record = dict(run_id=self.run_id, phase=phase, epoch=epoch,
                      micro_step=micro_step, optimizer_step=optimizer_step, **metrics)
        line = json.dumps(record, allow_nan=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Close on every record so completed steps survive later exceptions.
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
