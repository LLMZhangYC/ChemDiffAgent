#!/usr/bin/env python3
"""Replay saved tool calls without loading the language model or changing predictions."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def collect_calls(rows, source='chain', missing_only=False):
    from prepare_tool_retry import MISSING_MARKERS
    from rollout import TOOL_RE, _parse_payload
    calls = {}
    parse_errors = []
    for row_index, row in enumerate(rows):
        chain = row.get('tool_use_chain') or []
        candidates = []
        if source == 'chain' and chain:
            candidates = [(call.get('turn'), call.get('tool'), call.get('arguments'), call)
                          for call in chain if call.get('success') is False]
        else:
            by_turn = {call.get('turn'): call for call in chain}
            for turn, raw in enumerate(row.get('raw_generations') or [], 1):
                previous = by_turn.get(turn)
                if previous and previous.get('success') is True:
                    continue
                for match in TOOL_RE.finditer(raw):
                    try:
                        name, arguments = _parse_payload(match.group(1))
                        candidates.append((turn, name, arguments, previous))
                    except ValueError as exc:
                        parse_errors.append(dict(row_index=row_index, turn=turn, error=str(exc)))
        for turn, name, arguments, previous in candidates:
            if missing_only and (not previous or not any(
                    marker in str(previous.get('output', '')).lower() for marker in MISSING_MARKERS)):
                continue
            if not isinstance(name, str) or not isinstance(arguments, dict):
                parse_errors.append(dict(row_index=row_index, turn=turn, error='Invalid saved tool name/arguments'))
                continue
            key = (name, json.dumps(arguments, sort_keys=True, ensure_ascii=False))
            if key not in calls:
                calls[key] = dict(call_id=len(calls), tool=name, arguments=arguments, references=[])
            calls[key]['references'].append(dict(row_index=row_index, sample_id=row.get('sample_id'),
                                                roll_idx=row.get('roll_idx', 0), turn=turn,
                                                previous_output=previous.get('output') if previous else None))
    return list(calls.values()), parse_errors


def execute_calls(calls, executor, destination):
    counts = dict(unique_calls=len(calls), succeeded=0, failed=0)
    with destination.open('x', encoding='utf-8') as stream:
        for call in calls:
            started = time.perf_counter()
            output, success = executor.execute(call['tool'], call['arguments'])
            result = dict(call, output=output, success=success, seconds=time.perf_counter() - started)
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            stream.flush()
            counts['succeeded' if success else 'failed'] += 1
            print(f"[{call['call_id'] + 1}/{len(calls)}] {call['tool']}: {'OK' if success else output}", flush=True)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--config', type=Path, help='Apply runtime.env from inference YAML')
    parser.add_argument('--source', choices=['chain', 'raw'], default='chain')
    parser.add_argument('--missing-only', action='store_true')
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.results = args.results.resolve()
    if args.config:
        args.config = args.config.resolve()
    config_path = args.config or args.results.parent / 'config.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8')) if config_path.is_file() else {}
    if args.config and not args.config.is_file():
        raise FileNotFoundError(args.config)
    for key, value in config.get('runtime', {}).get('env', {}).items():
        os.environ[key] = str(value)
    os.environ.setdefault('RXN_T5CHEM_DEVICE', 'cpu')
    os.environ.setdefault('UNICORE_ROOT', str(ROOT / 'third_party/Uni-Core'))
    rows = [json.loads(line) for line in args.results.read_text(encoding='utf-8').splitlines() if line.strip()]
    # Resolve config paths as the inference launcher does, before importing tools.
    directory = Path.cwd() / 'res' / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '-tool_replay')
    os.chdir(ROOT)
    calls, errors = collect_calls(rows, args.source, args.missing_only)
    print(f'{sum(len(c["references"]) for c in calls)} saved calls -> {len(calls)} unique executions; '
          f'{len(errors)} parse errors')
    directory.mkdir(parents=True, exist_ok=False)
    plan = dict(source=str(args.results.resolve()), config=str(config_path.resolve()),
                mode=args.source, missing_only=args.missing_only, calls=calls, parse_errors=errors)
    (directory / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Output: {directory}')
    if args.dry_run:
        return
    from tool_registry import ToolExecutor, load_tools
    executor = ToolExecutor(load_tools(), timeout_seconds=args.timeout)
    counts = execute_calls(calls, executor, directory / 'tool_results.jsonl')
    (directory / 'summary.json').write_text(json.dumps(counts, indent=2), encoding='utf-8')
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
