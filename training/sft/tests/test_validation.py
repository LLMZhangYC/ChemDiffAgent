"""CPU tests for holdout isolation, fixed corruption and no-update evaluation."""
import ast
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.validation import evaluate, split_training_data
from train.metrics import MetricsLogger
from train.profiling import DisabledProfiler


class SplitTests(unittest.TestCase):
    def test_reproducible_disjoint_prompt_groups_and_no_rng_side_effect(self):
        rows = [{"prompt": f"q{i}", "response": f"a{j}", "step_map": [j]}
                for i in range(10) for j in range(2)]
        state = random.getstate()
        train, val, report = split_training_data(rows, 0.2, 17)
        self.assertEqual(state, random.getstate())
        self.assertEqual((len(train), len(val)), (16, 4))
        self.assertFalse({r["prompt"] for r in train} & {r["prompt"] for r in val})
        self.assertEqual(sorted(report["train_indices"] + report["validation_indices"]), list(range(20)))
        self.assertEqual(val, split_training_data(rows, 0.2, 17)[1])
        self.assertNotEqual(val, split_training_data(rows, 0.2, 18)[1])
        # Row ordering does not change held-out task membership.
        reversed_val = split_training_data(list(reversed(rows)), 0.2, 17)[1]
        self.assertEqual({r["prompt"] for r in val}, {r["prompt"] for r in reversed_val})
        self.assertEqual([r["step_map"] for r in train], [rows[i]["step_map"] for i in report["train_indices"]])

    def test_disabled_and_invalid_splits(self):
        rows = [{"prompt": "same", "response": "a"}]*3
        self.assertEqual(split_training_data(rows, 0)[0], rows)
        self.assertEqual(split_training_data(rows, 0)[1], [])
        for fraction in (-0.1, 1, True, float("nan")):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                split_training_data(rows, fraction)
        with self.assertRaisesRegex(ValueError, "distinct prompts"):
            split_training_data(rows, 0.1)
        with self.assertRaises(ValueError):
            split_training_data([], 0.1)


class EvaluationTests(unittest.TestCase):
    def test_uneven_batches_duplicate_removal_and_no_updates(self):
        model = torch.nn.Linear(1, 1).train()
        before = copy.deepcopy(model.state_dict())
        batches = [{"x": torch.tensor([[1.], [2.]])}, {"x": torch.tensor([[3.], [3.]])}]
        remaining = [2, 1]  # Simulate Accelerate trimming the final padded example.
        def gather(metrics):
            return metrics[:remaining.pop(0)]
        accelerator = NS(device="cpu", gather_for_metrics=gather)
        def forward_metrics(x):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            model(x)  # Run a real parameterized forward in the no-grad context.
            n = x[:, 0]
            return torch.stack([n, n*2, n, torch.ones_like(n)], dim=1)
        result = evaluate(model, batches, forward_metrics, accelerator)
        self.assertEqual(result["samples"], 3)
        self.assertEqual(result["loss"], 2)
        self.assertEqual(result["masked_cross_entropy"], 2)
        self.assertEqual(result["masked_tokens"], 6)
        self.assertEqual(result["masked_accuracy"], 0.5)
        self.assertTrue(model.training)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_error_restores_mode_and_empty_validation_fails(self):
        model = torch.nn.Linear(1, 1).train()
        accelerator = NS(device="cpu", gather_for_metrics=lambda x: x)
        def fail(**batch):
            raise RuntimeError("forward failed")
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            evaluate(model, [{"x": torch.ones(1)}], fail, accelerator)
        self.assertTrue(model.training)
        with self.assertRaisesRegex(ValueError, "no supervised"):
            evaluate(model, [], fail, accelerator)
        model.eval()
        with self.assertRaises(ValueError):
            evaluate(model, [], fail, accelerator)
        self.assertFalse(model.training)


class ActualSFTFunctionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "train/sft_sdar.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        names = {"collect_training_data", "find_token_positions", "forward_process", "process_pad",
                 "collapse_k_unique", "one_round_vectorized", "make_basic_block_attention"}
        functions = [n for n in main.body if isinstance(n, ast.FunctionDef) and n.name in names]
        cls.code = compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec")

    def environment(self):
        class Settings(dict):
            __getattr__ = dict.__getitem__
        config = NS(training=Settings(method="semi-ar", block_size=4, lower_p=.1, upper_p=.9,
                                      mask_tool_responses=True))
        env = dict(torch=torch, F=F, config=config, start_pos=2, L0=2, L1=16,
                   mask_id=2, pad_id=0, post_num=0,
                   tokenizer=NS(convert_tokens_to_ids=lambda tag: 3 if tag == "<tool_response>" else 4))
        exec(self.code, env)
        return env

    def test_fixed_validation_corruption_preserves_rng_and_tool_exclusion(self):
        env = self.environment()
        response = [5, 6, 3, 8, 8, 4] + [7]*9 + [0]
        ids = torch.tensor([[5, 6]+response]*3)
        rng = torch.random.get_rng_state()
        first = env["collect_training_data"](ids, [], generator=torch.Generator().manual_seed(91))
        second = env["collect_training_data"](ids, [], generator=torch.Generator().manual_seed(91))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        extended, mask, positions, labels = first
        self.assertFalse(mask[:, :2].any())
        self.assertFalse(mask[:, 4:8].any())  # tool tags and content
        self.assertFalse(mask[:, -1].any())  # response padding
        self.assertTrue(torch.equal(extended[:, 18+2:18+6], ids[:, 4:8]))
        self.assertTrue(torch.equal(positions[:, 18:], positions[:, 2:18]))

    def test_forward_metrics_match_training_loss_and_masked_accuracy(self):
        env = self.environment()
        ids = torch.tensor([[5, 6]+[7]*16, [5, 6]+[8]*16])
        tensors = env["collect_training_data"](ids, [], generator=torch.Generator().manual_seed(19))
        embedding, head = torch.nn.Embedding(16, 8), torch.nn.Linear(8, 16)
        def model(input_ids, **kw):
            return NS(logits=head(embedding(input_ids)))
        env["model"] = model
        env["basic_block_attention"] = env["make_basic_block_attention"](34, 2, 4)
        batch = dict(zip(["extended_input_ids", "p_mask", "tok_idx_ext", "labels"], tensors))
        loss = env["forward_process"](**batch)
        metrics = env["forward_process"](**batch, return_metrics=True)
        self.assertTrue(torch.allclose(loss, metrics[:, 0].mean()))
        self.assertTrue(torch.equal(metrics[:, 2], tensors[1].sum(1).float()))
        loss.backward()
        self.assertIsNotNone(head.weight.grad)

    def test_periodic_validation_uses_update_boundaries_and_epoch_end(self):
        source = Path(__file__).resolve().parents[1] / "train/sft_sdar.py"
        main = next(n for n in ast.parse(source.read_text(encoding="utf-8")).body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        run_validation = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "run_validation")
        loop = next(n for n in main.body if isinstance(n, ast.For) and n.target.id == "epoch")
        setup = ast.parse("global_steps=0\ncompleted_optimizer_steps=0\nlast_validation_step=None\nlast_validation_epoch=None\nvalidation_history=[]\n").body
        finish = ast.parse("return validation_history\n").body
        function = ast.FunctionDef(name="exercise", args=ast.arguments(posonlyargs=[], args=[],
                                   kwonlyargs=[], kw_defaults=[], defaults=[]),
                                   body=setup+[run_validation, loop]+finish, decorator_list=[])
        code = compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(source), "exec")
        class Progress(list):
            def set_postfix(self, **kw): pass
        for interval, after_epoch, enabled, cap, expected in [(3, True, True, None, [2, 3, 4]),
                                                         (1, True, True, None, [1, 2, 3, 4]),
                                                         (3, False, True, None, [3]),
                                                         (1, True, False, None, []),
                                                         (3, True, True, 3, [3])]:
            with self.subTest(interval=interval, epoch=after_epoch, enabled=enabled), tempfile.TemporaryDirectory() as directory:
                model = torch.nn.Linear(1, 1)
                model.global_steps = 0
                micro = [0]
                def backward(loss):
                    micro[0] += 1
                    if micro[0] % 2 == 0:
                        model.global_steps += 1
                config = NS(get=lambda *a: {}, training=NS(max_grad_norm=None),
                            model=NS(optimized_name="optimized"))
                batch = {name: torch.ones(1) for name in ["extended_input_ids", "p_mask", "tok_idx_ext", "labels"]}
                namespace = dict(first_epoch=0, num_train_epochs=2, model=model, config=config,
                    torch=torch, json=json, Path=Path, output_dir=directory, tokenizer=None,
                    metrics_logger=MetricsLogger(directory), validation_enabled=enabled, validation_config={"eval_after_epoch": after_epoch},
                    eval_steps=interval, validation_dataloader=[], max_optimizer_steps=cap,
                    create_profiler=lambda *a, **kw: DisabledProfiler(),
                    evaluate=lambda *a: {"loss": 1, "masked_accuracy": .5},
                    forward_process=lambda **kw: torch.tensor(1.),
                    tqdm=lambda data, **kw: Progress(data), train_dataloader_lm=[batch]*(8 if cap else 4),
                    data_time_m=NS(update=lambda x: None), time=NS(time=lambda: 0), end=0,
                    optimizer=NS(step=lambda: None, zero_grad=lambda **kw: None, param_groups=[{"lr": 1e-5}]),
                    lr_scheduler=NS(step=lambda: None), logger=NS(info=lambda x: None),
                    save_checkpoint=lambda *a, **kw: None,
                    accelerator=NS(device=torch.device("cpu"), process_index=0, num_processes=1,
                                   is_local_main_process=True, is_main_process=True,
                                   backward=backward, reduce=lambda value, **kw: value, wait_for_everyone=lambda: None))
                exec(code, namespace)
                history = namespace["exercise"]()
                self.assertEqual([row["optimizer_step"] for row in history], expected)
                logged = [json.loads(line) for line in (Path(directory)/"metrics.jsonl").read_text().splitlines()]
                training_rows = [r for r in logged if r["phase"] == "train"]
                validation_rows = [r for r in logged if r["phase"] == "validation"]
                self.assertEqual(len(training_rows), 6 if cap else 8)
                self.assertEqual([r["micro_step"] for r in training_rows], list(range(1, len(training_rows)+1)))
                self.assertEqual([r["optimizer_step"] for r in validation_rows], expected)
                if expected:
                    lines = (Path(directory)/"validation_metrics.jsonl").read_text().splitlines()
                    self.assertEqual(len(lines), len(expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)
