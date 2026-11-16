#!/usr/bin/env python3
"""Prepare a separate inference dataset/config for missing-tool failures."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import yaml


MISSING_MARKERS = ('toolbackendunavailable', 'modulenotfounderror', 'importerror',
                   'filenotfounderror', 'no module named', 'checkpoint is missing',
                   'checkpoint is also missing', 'source is missing', 'no such file',
                   'requires uni-core', 'vocab is missing')


def select_samples(rows, all_tool_errors=False):
    selected = {}
    for row in rows:
        failed = [call for call in row.get('tool_use_chain', []) if call.get('success') is False]
        matches = failed if all_tool_errors else [
            call for call in failed if any(marker in str(call.get('output', '')).lower()
                                          for marker in MISSING_MARKERS)]
        if not matches:
            continue
        original = row.get('original_data')
        if not isinstance(original, dict) or 'question' not in original:
            raise ValueError('Selected result has no original_data.question; cannot reconstruct input')
        sample = dict(original)
        sample.setdefault('sample_id', row.get('sample_id'))
        # Retry the whole sample, once, even if several tool calls/rollouts failed.
        key = (str(sample.get('sample_id')), sample['question'])
        selected[key] = sample
    return list(selected.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, help='Previous single_test.jsonl results')
    parser.add_argument('--config', type=Path, help='Defaults to config.yaml beside results')
    parser.add_argument('--all-tool-errors', action='store_true', help='Include all failed tool calls')
    args = parser.parse_args()
    with args.results.open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    samples = select_samples(rows, args.all_tool_errors)
    print(f'Selected {len(samples)} samples from {len(rows)} completed rollouts')
    if not samples:
        return
    source = args.config or args.results.parent / 'config.yaml'
    config = yaml.safe_load(source.read_text(encoding='utf-8'))
    directory = args.results.resolve().parent / ('tool_retry_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(exist_ok=False)
    dataset = directory / 'single_test.jsonl'
    dataset.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in samples), encoding='utf-8')
    config['data'] = {'directory': str(directory), 'datasets': ['single_test']}
    config.setdefault('inference', {}).update(max_samples=0, sample_id='', num_rolls=1,
                                            num_shards=1, shard_index=0, resume=False)
    config.setdefault('profiling', {})['enabled'] = False
    config_path = directory / 'inference_npu_tool_retry.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
    print(f'Retry config: {config_path}')
    print(f'bash scripts/run_inference_npu.sh "{config_path}"')


if __name__ == '__main__':
    main()
