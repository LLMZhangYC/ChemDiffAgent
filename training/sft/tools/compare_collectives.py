#!/usr/bin/env python3
"""Compare recorded DeepSpeed host call signatures; needs only Python stdlib."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def signature(record):
    def strip(value):
        if isinstance(value, list):
            return [strip(x) for x in value]
        if isinstance(value, dict):
            # Pointers/devices/storage offsets legitimately differ by rank.
            return {k: strip(v) for k, v in value.items() if k not in ("ptr", "device", "storage_offset")}
        return value
    return json.dumps({k: strip(record.get(k)) for k in ("op", "tensors", "options")}, sort_keys=True)


def compare(records):
    groups = defaultdict(lambda: defaultdict(dict))
    for record in records:
        if record.get("event") == "enter":
            key = json.dumps(record["group"], sort_keys=True)
            rank, seq = record["rank"], record["sequence"]
            if seq in groups[key][rank]:
                raise ValueError(f"Duplicate sequence {seq} on rank {rank}; do not mix runs")
            groups[key][rank][seq] = record
    results = []
    for key, by_rank in groups.items():
        group = json.loads(key)
        expected = group["ranks"]
        present = sorted(by_rank)
        common = set.intersection(*(set(by_rank.get(r, {})) for r in expected))
        mismatch = None
        for seq in sorted(common):
            rows = [by_rank[r][seq] for r in expected]
            if len({signature(row) for row in rows}) != 1:
                mismatch = {"sequence": seq, "calls": rows}
                break
        missing = []
        for seq in sorted(set.union(*(set(rows) for rows in by_rank.values()))):
            absent = [r for r in expected if seq not in by_rank.get(r, {})]
            if absent:
                missing.append({"sequence": seq, "missing_ranks": absent})
        results.append(dict(group=group, present_ranks=present, common_calls=len(common),
                            first_signature_mismatch=mismatch, missing_sequence_count=len(missing),
                            missing_sequence_examples=missing[:8],
                            last_enter={str(r): rows[max(rows)] for r, rows in by_rank.items()}))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="This run's output/diagnostics directory")
    args = parser.parse_args()
    files = sorted(args.directory.glob("collectives_rank_*.jsonl"))
    if not files:
        parser.error("No collective logs; run with diagnostics.trace_collectives=true first")
    records, errors, invalid = [], [], []
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                except ValueError:
                    invalid.append(f"{path.name}:{line_number}")
                    continue
                if record.get("event") == "enter":
                    records.append(record)
                elif record.get("event") in ("error", "fetch_error"):
                    errors.append(record)
    if not records:
        parser.error("No recorded entries; the run may not have reached collectives_start_micro_step")
    print(json.dumps(dict(
        scope="DeepSpeed host entries only; sequence is NOT HCCL opIndex",
        interpretation="Matching signatures do not prove device completion. Missing tails can follow another rank's failure; not proof of a missing dispatch. Compare CANN entry/first-error logs too.",
        invalid_lines=invalid, groups=compare(records), errors=errors), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
