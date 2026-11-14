"""CPU tests: real safetensors/tokenizer, no public weights or accelerator needed."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import SDARConfig, SDARForCausalLM
from tools import export_reduced_sdar as export
from train.prompting_utils import UniversalPrompting


def tokenizer():
    vocab = {name: i for i, name in enumerate([
        "[PAD]", "[UNK]", "[MASK]", "<tool_response>", "</tool_response>",
        "p", "r", "tool", "result", "[EOS]", "a", "b", "c", "d", "e", "f"])}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                                  mask_token="[MASK]", eos_token="[EOS]",
                                  additional_special_tokens=["<tool_response>", "</tool_response>"])


def tiny_config():
    return SDARConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=16,
                      pad_token_id=0, eos_token_id=9, max_position_embeddings=8192,
                      use_cache=False, use_sliding_window=False, sliding_window=None,
                      fuse_cross_entropy=False)


class TruncationTests(unittest.TestCase):
    def setUp(self):
        self.tok = tokenizer()

    def test_budget_uses_padded_prompt_and_preserves_sample_alignment(self):
        formatter = UniversalPrompting(self.tok, max_prompt_len=6, max_gen_length=99, max_seq_length=18)
        ids, labels, p, dropped = formatter((["p " * 2, "p " * 7, "p " * 6], ["r " * 40] * 3))
        self.assertEqual((p, ids.shape[1], dropped), (6, 12, 1))
        self.assertEqual(formatter.kept_indices, [0, 2])
        self.assertEqual(p + 2 * (ids.shape[1] - p), 18)
        self.assertEqual(ids[0, :4].tolist(), [0] * 4)
        torch.testing.assert_close(ids, labels)

    def test_incomplete_tool_span_is_removed(self):
        formatter = UniversalPrompting(self.tok, max_prompt_len=4, max_gen_length=99, max_seq_length=16)
        ids, _, p, _ = formatter((["p " * 4], ["r r <tool_response> tool result tool result </tool_response> r"]))
        self.assertEqual(ids[0, p:].tolist(), [6, 6])
        self.assertEqual(formatter.stats["trimmed_tool_spans"], 1)

    def test_closed_tool_span_is_retained(self):
        formatter = UniversalPrompting(self.tok, max_prompt_len=4, max_gen_length=99, max_seq_length=20)
        ids, _, p, _ = formatter((["p p"], ["r <tool_response> tool </tool_response> r"]))
        self.assertEqual(ids[0, p:].tolist(), [6, 3, 7, 4, 6])

    def test_no_target_sample_filtered_with_original_index(self):
        formatter = UniversalPrompting(self.tok, max_prompt_len=4, max_gen_length=99, max_seq_length=20)
        _, _, _, dropped = formatter((["p", "p"], ["<tool_response> tool </tool_response>", "r r"]))
        self.assertEqual(formatter.kept_indices, [1])
        self.assertEqual(dropped, 1)

    def test_exact_8192_and_short_sample_padding(self):
        formatter = UniversalPrompting(self.tok, max_prompt_len=4096, max_gen_length=8192, max_seq_length=8192)
        ids, _, p, _ = formatter((["p " * 256, "p"], ["r " * 9000, "r"]))
        self.assertEqual(p + 2 * (ids.shape[1] - p), 8192)
        self.assertEqual(formatter.stats["expanded_length"], 8192)
        self.assertEqual(int(ids[1, p:].ne(0).sum()), 1)

    def test_no_valid_data_and_malformed_tags_fail(self):
        formatter = UniversalPrompting(self.tok, max_prompt_len=2, max_gen_length=99, max_seq_length=16)
        with self.assertRaises(ValueError):
            formatter((["p p p"], ["r"]))
        with self.assertRaises(ValueError):
            formatter((["p"], ["</tool_response> r"]))
        with self.assertRaises(ValueError):
            formatter((["p"], [""]))

    def test_legacy_path_unchanged_without_total_limit(self):
        formatter = UniversalPrompting(self.tok, max_prompt_len=4, max_gen_length=2)
        ids, _, p, dropped = formatter((["p", "p p"], ["r r r", "r"]))
        self.assertEqual((p, dropped), (2, 0))
        self.assertEqual(ids.tolist(), [[0, 5, 6, 6], [5, 5, 6, 0]])


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        torch.manual_seed(123)
        self.model = SDARForCausalLM(tiny_config())
        self.state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.config.save_pretrained(self.source)
        tokenizer().save_pretrained(self.source)

    def tearDown(self):
        self.temp.cleanup()

    def test_sharded_export_reads_only_selected_tensors(self):
        selected = {k: v for k, v in self.state.items() if not k.startswith("model.layers.") or k.startswith("model.layers.0.")}
        mixed = {**selected, "model.layers.1.self_attn.q_proj.weight": self.state["model.layers.1.self_attn.q_proj.weight"]}
        save_file(mixed, str(self.source / "mixed.safetensors"))
        # The later-layer shard deliberately does not exist: exporting layer 0
        # must not touch it, even though its keys exist in the source index.
        weight_map = {k: "mixed.safetensors" if k in mixed else "DO_NOT_OPEN.safetensors" for k in self.state}
        (self.source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        reads = []
        class RecordingReader:
            def __init__(self, *args, **kwargs):
                self.context = safe_open(*args, **kwargs)
            def __enter__(self):
                self.handle = self.context.__enter__()
                return self
            def __exit__(self, *args):
                return self.context.__exit__(*args)
            def get_slice(self, key):
                return self.handle.get_slice(key)
            def get_tensor(self, key):
                reads.append(key)
                return self.handle.get_tensor(key)
        output = self.root / "reduced"
        with patch.object(export, "safe_open", RecordingReader):
            report = export.export_prefix(self.source, output, 1)
        self.assertEqual(set(reads), set(selected))
        self.assertEqual(report["num_hidden_layers"], 1)
        self.assertEqual(report["read_shards"], ["mixed.safetensors"])
        loaded, info = SDARForCausalLM.from_pretrained(output, output_loading_info=True)
        self.assertFalse(any(info.values()), info)
        self.assertEqual(len(loaded.model.layers), 1)
        self.assertEqual(set(loaded.state_dict()), set(selected))
        for key, tensor in loaded.state_dict().items():
            torch.testing.assert_close(tensor, selected[key], rtol=0, atol=0)
        self.model.config.num_hidden_layers = 1
        ids = torch.tensor([[5, 6, 7, 8]])
        attention = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        with torch.no_grad():
            expected = self.model.eval()(input_ids=ids, attention_mask=attention).logits
            actual = loaded.eval()(input_ids=ids, attention_mask=attention).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            export.export_prefix(self.source, output, 1)

    def test_single_file_export_and_depth_validation(self):
        save_file(self.state, str(self.source / "model.safetensors"))
        report = export.export_prefix(self.source, self.root / "single", 1)
        self.assertEqual(report["weight_tensors"], 14)
        reloaded = AutoModelForCausalLM.from_pretrained(
            self.root / "single", trust_remote_code=True, local_files_only=True)
        self.assertEqual(len(reloaded.model.layers), 1)
        for depth in [0, 4, True, 1.5]:
            with self.assertRaises(ValueError):
                export.export_prefix(self.source, self.root / "invalid", depth)

    def test_wrong_tensor_shape_fails_atomically(self):
        self.state["model.layers.0.self_attn.q_proj.weight"] = torch.zeros(1, 1)
        save_file(self.state, str(self.source / "model.safetensors"))
        output = self.root / "invalid"
        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            export.export_prefix(self.source, output, 1)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
