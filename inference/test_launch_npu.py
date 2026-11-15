"""Validate YAML-to-process arguments without NPU or a model."""
import unittest
import sys
from pathlib import Path
import yaml
from launch_npu import build_commands
from launch_npu import run_directory
from datetime import datetime


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(
            Path(__file__).with_name('configs').joinpath('inference_npu.yaml').read_text())

    def test_example_and_multiple_datasets(self):
        self.config['data']['datasets'].append('multi_test')
        commands = build_commands(self.config)
        self.assertEqual(len(commands), 2)
        for command, dataset in zip(commands, ['single_test', 'multi_test']):
            self.assertEqual(command[0], 'bash')
            self.assertIn('--profile', command)
            self.assertEqual(Path(command[command.index('--profile-dir') + 1]), Path('prof') / dataset)
            self.assertIn('--print-turns', command)
            self.assertNotIn('--resume', command)
            self.assertIn('npu:0', command)

    def test_disabled_profile_and_existing_environment(self):
        self.config['runtime']['cann_env'] = None
        self.config['profiling']['enabled'] = False
        command = build_commands(self.config)[0]
        self.assertEqual(command[0], self.config['runtime'].get('python', sys.executable))
        self.assertNotIn('--profile', command)

    def test_typo_is_rejected(self):
        self.config['inference']['max_token'] = 1
        with self.assertRaises(ValueError):
            build_commands(self.config)

    def test_output_is_resolved_before_changing_subprocess_directory(self):
        directory = run_directory('configs/custom.yaml', datetime(2026, 11, 15, 12, 30, 45))
        command = build_commands(self.config, directory)[0]
        self.assertEqual(Path(command[command.index('--output') + 1]),
                         Path.cwd() / 'res/20261115_123045_000000-custom/single_test.jsonl')


if __name__ == '__main__':
    unittest.main()
