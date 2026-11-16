import unittest
from prepare_tool_retry import select_samples


class RetryTests(unittest.TestCase):
    def row(self, success, output, sample='a'):
        return dict(sample_id=sample, original_data=dict(sample_id=sample, question='question'),
                    tool_use_chain=[dict(success=success, output=output)])

    def test_select_missing_only_and_deduplicate(self):
        rows = [self.row(False, "Error: ModuleNotFoundError: No module named 'rdkit'"),
                self.row(False, 'Error: ToolBackendUnavailable: checkpoint is missing'),
                self.row(True, 'ok', 'b'), self.row(False, 'Error: TimeoutError', 'c')]
        self.assertEqual(len(select_samples(rows)), 1)
        self.assertEqual(len(select_samples(rows, all_tool_errors=True)), 2)

    def test_does_not_retry_format_errors_or_empty_chains(self):
        self.assertEqual(select_samples([dict(termination_reason='invalid_generation', tool_use_chain=[])]), [])

    def test_missing_original_is_rejected(self):
        row = self.row(False, 'Error: FileNotFoundError')
        del row['original_data']
        with self.assertRaises(ValueError):
            select_samples([row])


if __name__ == '__main__':
    unittest.main()
