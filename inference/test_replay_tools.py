import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from replay_tools import collect_calls, execute_calls


class ReplayTests(unittest.TestCase):
    def row(self, success=False):
        return dict(sample_id='test', tool_use_chain=[dict(turn=1, tool='TestMolecule',
                    arguments={'input': 'CCO'}, success=success,
                    output='Error: ModuleNotFoundError: rdkit')],
                    raw_generations=['<tool_call>{"name":"TestMolecule","arguments":{"input":"CCO"}}</tool_call>'])

    def test_dedup_and_skip_success(self):
        calls, errors = collect_calls([self.row(), self.row(), self.row(True)])
        self.assertEqual(errors, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]['references']), 2)

    def test_raw_parsing_and_missing_filter(self):
        calls, errors = collect_calls([self.row()], source='raw', missing_only=True)
        self.assertEqual(len(calls), 1)
        row = self.row()
        row['tool_use_chain'] = []
        self.assertEqual(len(collect_calls([row])[0]), 1)
        self.assertEqual(collect_calls([row], missing_only=True)[0], [])
        row['raw_generations'] = ['<tool_call>broken</tool_call>', 'ordinary answer']
        self.assertEqual(len(collect_calls([row])[1]), 1)

    def test_execution_persists_results_without_mutating_old_calls(self):
        calls, _ = collect_calls([self.row()])
        executor = Mock()
        executor.execute.return_value = ('new result', True)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'tools.jsonl'
            summary = execute_calls(calls, executor, path)
            result = json.loads(path.read_text())
        executor.execute.assert_called_once_with('TestMolecule', {'input': 'CCO'})
        self.assertEqual(summary['succeeded'], 1)
        self.assertEqual(result['output'], 'new result')
        self.assertNotIn('output', calls[0])


if __name__ == '__main__':
    unittest.main()
