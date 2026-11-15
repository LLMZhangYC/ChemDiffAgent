"""CPU contract tests with mocked torch_npu; NOT native NPU kernel validation.

Extract the actual operator definitions so these tests need only torch, without
the full training environment or a particular Transformers version.
"""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
import torch.nn.functional as F


source = Path(__file__).resolve().parents[1] / "models/sdar/modeling_sdar.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
               and node.name in {"SDARRMSNorm", "SDARMLP", "npu_attention_forward"}]
required = {"SDARRMSNorm": "npu_rms_norm", "SDARMLP": "npu_swiglu",
            "npu_attention_forward": "npu_fusion_attention"}
by_name = {node.name: node for node in definitions}
missing = [f"{name} -> {operator}" for name, operator in required.items()
           if name not in by_name or not any(
               isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == operator for node in ast.walk(by_name[name]))]
if missing:
    raise RuntimeError(
        f"Native NPU implementation missing in {source}: {', '.join(missing)}. "
        "Sync models/sdar/modeling_sdar.py and this test from the same npu-dev revision; "
        "copying the test alone is insufficient. This is a source-version mismatch, "
        "not a native NPU kernel failure.")
for node in definitions:
    if isinstance(node, ast.ClassDef):
        node.decorator_list = []
namespace = {"torch": torch, "nn": nn, "F": F}
exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])),
             str(source), "exec"), namespace)


def assert_close(actual, expected):
    if actual.shape != expected.shape or not torch.allclose(actual, expected, rtol=1e-5, atol=1e-6):
        raise AssertionError("Operator contract tensors differ")


class OperatorContracts(unittest.TestCase):
    def test_masked_gqa_outputs_and_gradients(self):
        torch.manual_seed(7)
        for kind in ("bool", "float", "padding", "none"):
            with self.subTest(mask=kind):
                q = torch.randn(2, 4, 6, 8, requires_grad=True)
                k = torch.randn(2, 2, 6, 8, requires_grad=True)
                v = torch.randn(2, 2, 6, 8, requires_grad=True)
                visible = torch.eye(6, dtype=torch.bool)[None, None].repeat(2, 1, 1, 1)
                visible[..., :2] = True
                visible[..., 2:4, 2:4] = True
                visible[..., 4:, 4:] = True
                mask = visible.float() if kind == "float" else visible
                if kind == "padding":
                    mask = torch.tensor([[1, 1, 0, 1, 0, 1]] * 2)
                    visible = mask[:, None, None, :].bool().expand(2, 1, 6, 6)
                elif kind == "none":
                    mask = visible = None

                def native(query, key, value, **kwargs):
                    self.assertEqual(kwargs["input_layout"], "BNSD")
                    self.assertEqual(kwargs["head_num"], 4)
                    self.assertEqual(key.shape[1], 2)  # no premature GQA expansion
                    self.assertEqual(kwargs["sparse_mode"], 0)
                    self.assertTrue(all(x.is_contiguous() for x in (query, key, value)))
                    blocked = kwargs["atten_mask"]
                    if visible is None:
                        self.assertIsNone(blocked)
                    else:
                        assert_close(blocked, ~visible)
                    output = F.scaled_dot_product_attention(
                        query, key.repeat_interleave(2, 1), value.repeat_interleave(2, 1),
                        attn_mask=None if blocked is None else ~blocked, scale=kwargs["scale"])
                    return (output,)

                backend = SimpleNamespace(npu_fusion_attention=Mock(side_effect=native))
                with patch.dict(sys.modules, {"torch_npu": backend}):
                    actual = namespace["npu_attention_forward"](q, k, v, mask, 8 ** -0.5)
                expected = F.scaled_dot_product_attention(
                    q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1),
                    attn_mask=visible, scale=8 ** -0.5)
                assert_close(actual, expected)
                grad = torch.randn_like(actual)
                actual_grads = torch.autograd.grad(actual, (q, k, v), grad, retain_graph=True)
                expected_grads = torch.autograd.grad(expected, (q, k, v), grad)
                for a, b in zip(actual_grads, expected_grads):
                    assert_close(a, b)
                backend.npu_fusion_attention.assert_called_once()

    def test_rmsnorm_npu_dispatch(self):
        x = torch.randn(2, 3, 8, requires_grad=True)
        weight = nn.Parameter(torch.randn(8))
        backend = SimpleNamespace(npu_rms_norm=Mock(side_effect=lambda a, w, eps:
                                  (F.rms_norm(a, (8,), w, eps), None)))
        wrapped = SimpleNamespace(device=SimpleNamespace(type="npu"), contiguous=lambda: x)
        with patch.dict(sys.modules, {"torch_npu": backend}):
            actual = namespace["SDARRMSNorm"].forward(
                SimpleNamespace(weight=weight, variance_epsilon=1e-6), wrapped)
        assert_close(actual, F.rms_norm(x, (8,), weight, 1e-6))
        actual.sum().backward()
        self.assertIsNotNone(weight.grad)
        backend.npu_rms_norm.assert_called_once()

    def test_swiglu_gate_order_and_gradients(self):
        x = torch.randn(2, 3, 8, requires_grad=True)
        gate, up, down = nn.Linear(8, 16), nn.Linear(8, 16), nn.Linear(16, 8)
        wrapped = SimpleNamespace(device=SimpleNamespace(type="npu"))
        def swiglu(packed, dim):
            a, b = packed.chunk(2, dim)
            assert_close(a, gate(x))
            assert_close(b, up(x))
            return F.silu(a) * b
        backend = SimpleNamespace(npu_swiglu=Mock(side_effect=swiglu))
        mlp = SimpleNamespace(config=SimpleNamespace(hidden_act="silu"),
                              gate_proj=lambda _: gate(x), up_proj=lambda _: up(x), down_proj=down)
        with patch.dict(sys.modules, {"torch_npu": backend}):
            actual = namespace["SDARMLP"].forward(mlp, wrapped)
        expected = down(F.silu(gate(x)) * up(x))
        assert_close(actual, expected)
        actual.sum().backward()
        self.assertTrue(all(p.grad is not None for layer in (gate, up, down) for p in layer.parameters()))
        backend.npu_swiglu.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
