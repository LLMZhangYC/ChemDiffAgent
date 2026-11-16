import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from replay_tools import collect_calls, execute_calls, merge_results


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

    def test_merge_preserves_all_samples_and_model_outputs(self):
        rows = [self.row(), self.row(True)]
        rows[0].update(prediction='old answer', messages=[{'content': 'old error'}],
                       evaluation={'all_tool_calls_succeeded': False})
        calls, _ = collect_calls(rows)
        result = dict(calls[0], output='fixed', success=True, seconds=0.1)
        merged = merge_results(rows, [result])
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[1], rows[1])
        self.assertEqual(merged[0]['tool_use_chain'][0]['output'], 'fixed')
        self.assertEqual(merged[0]['prediction'], 'old answer')
        self.assertEqual(merged[0]['messages'], rows[0]['messages'])
        self.assertFalse(merged[0]['tool_replay']['prediction_recomputed'])
        self.assertTrue(merged[0]['evaluation']['all_tool_calls_succeeded'])
        self.assertFalse(merged[0]['tool_replay']['original_evaluation']['all_tool_calls_succeeded'])
        self.assertFalse(rows[0]['tool_use_chain'][0]['success'])

    def test_merge_preserves_still_failed_calls_and_rejects_wrong_source(self):
        rows = [self.row()]
        calls, _ = collect_calls(rows)
        result = dict(calls[0], output='Invalid SMILES', success=False, seconds=0.2)
        merged = merge_results(rows, [result])
        self.assertFalse(merged[0]['tool_use_chain'][0]['success'])
        self.assertEqual(len(merged[0]['tool_use_chain'][0]['replay_history']), 1)
        rows[0]['sample_id'] = 'different'
        with self.assertRaises(ValueError):
            merge_results(rows, [result])

    def test_merge_can_recover_calls_from_raw(self):
        row = self.row()
        row['tool_use_chain'] = []
        calls, _ = collect_calls([row])
        merged = merge_results([row], [dict(calls[0], output='ok', success=True, seconds=0.1)])
        self.assertTrue(merged[0]['tool_use_chain'][0]['recovered_from_raw'])


if __name__ == '__main__':
    unittest.main()
