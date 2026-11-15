"""CPU contracts for inference profiling; no checkpoint or NPU required."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class ForwardProfilingTests(unittest.TestCase):
    def test_forward_steps_only_after_success(self):
        tree = ast.parse(Path(__file__).with_name('dllm_generation.py').read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_forward')
        namespace = {'nullcontext': nullcontext}
        exec(compile(ast.Module(body=[forward], type_ignores=[]), '<forward>', 'exec'), namespace)
        profiler = Mock(spec=['step'])
        owner = SimpleNamespace(profiler=profiler)
        model = Mock(return_value='output')
        self.assertEqual(namespace['_forward'](owner, model, input_ids=123), 'output')
        model.assert_called_once_with(input_ids=123)
        profiler.step.assert_called_once_with()
        model.side_effect = RuntimeError('forward failed')
        with self.assertRaises(RuntimeError):
            namespace['_forward'](owner, model)
        profiler.step.assert_called_once_with()
        owner.profiler = None
        model.side_effect = None
        namespace['_forward'](owner, model)

    def test_all_generation_forwards_are_profiled(self):
        tree = ast.parse(Path(__file__).with_name('dllm_generation.py').read_text(encoding='utf-8'))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        wrapped = [node for node in calls if isinstance(node.func, ast.Attribute)
                   and node.func.attr == '_forward']
        self.assertEqual(len(wrapped), 4)  # prefill, denoise, fallback, cache commit
        direct = [node for node in calls if isinstance(node.func, ast.Attribute)
                  and ast.unparse(node.func) in {'self.model', 'self.model.model'}]
        self.assertEqual(direct, [])


if __name__ == '__main__':
    unittest.main()
