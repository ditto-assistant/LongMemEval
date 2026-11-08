import functools
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.retrieval import run_retrieval


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        values = [1 if "red" in text else 0 for text in texts]
        tokens = torch.tensor([[v, v] for v in values])
        return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}


class FakeOutput:
    def __init__(self, hidden):
        self.last_hidden_state = hidden

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.last_hidden_state


class CacheSensitiveModel:
    device = torch.device("cpu")

    def __init__(self):
        self.calls = 0

    def __call__(self, input_ids, attention_mask, use_cache=None):
        if use_cache is not False:
            raise RuntimeError("legacy model attempted unsupported cache API")
        self.calls += 1
        return FakeOutput(torch.nn.functional.one_hot(input_ids, num_classes=2).float())


class LoadableModel(CacheSensitiveModel):
    config = SimpleNamespace(hidden_size=2)

    def to(self, device):
        return self

    def eval(self):
        return self


class Qwen2RotaryEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 8
        self.base = 1000000.0
        self.max_position_embeddings = 128
        self.max_seq_len_cached = 128
        self.register_buffer("inv_freq", torch.full((4,), -1.0), persistent=False)
        self.register_buffer("cos_cached", torch.full((128, 8), -1.0), persistent=False)
        self.register_buffer("sin_cached", torch.full((128, 8), -1.0), persistent=False)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        positions = torch.arange(seq_len, device=device, dtype=torch.int64).type_as(self.inv_freq)
        frequencies = torch.outer(positions, self.inv_freq)
        values = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos_cached", values.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", values.sin().to(dtype), persistent=False)

    def forward(self, length):
        if length > self.max_seq_len_cached:
            self._set_cos_sin_cache(length, self.inv_freq.device, torch.get_default_dtype())
        return self.cos_cached[:length], self.sin_cached[:length]


class RetrievalCompatibilityTest(unittest.TestCase):
    def test_legacy_qwen_tokenizer_respects_checkpoint_eos_setting(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(run_retrieval.AutoTokenizer, 'from_pretrained') as loader:
            config_path = Path(directory) / 'tokenizer_config.json'
            for value, expected in ((True, {'add_eos_token': True}),
                                    (False, {'add_eos_token': False}),
                                    (None, {})):
                with self.subTest(value=value):
                    config_path.write_text(json.dumps({} if value is None else {'add_eos_token': value}))
                    run_retrieval.load_legacy_qwen_tokenizer(directory)
                    loader.assert_called_with(directory, trust_remote_code=True, revision=None, **expected)
            config_path.unlink()
            run_retrieval.load_legacy_qwen_tokenizer(directory)
            loader.assert_called_with(directory, trust_remote_code=True, revision=None)
            config_path.write_text(json.dumps({'add_eos_token': 'true'}))
            with self.assertRaisesRegex(ValueError, 'must be a boolean'):
                run_retrieval.load_legacy_qwen_tokenizer(directory)

    def test_remote_tokenizer_uses_same_resolved_revision(self):
        source = 'example/model'
        revision = '0123456789abcdef0123456789abcdef01234567'
        with patch.object(run_retrieval, 'resolve_revision', return_value=revision), \
             patch.object(run_retrieval, 'get_tokenizer_config', return_value={'add_eos_token': True}) as config, \
             patch.object(run_retrieval.AutoTokenizer, 'from_pretrained') as loader:
            run_retrieval.load_legacy_qwen_tokenizer(source)
        config.assert_called_once_with(source, revision=revision)
        loader.assert_called_once_with(source, revision=revision, trust_remote_code=True, add_eos_token=True)

    def test_legacy_rope_base_is_preserved(self):
        config = SimpleNamespace(rope_parameters={"rope_theta": 1000000.0})
        with patch.object(run_retrieval.AutoConfig, "from_pretrained", return_value=config):
            self.assertIs(run_retrieval.load_legacy_qwen_config("fixture"), config)
        self.assertEqual(config.rope_theta, 1000000.0)

        existing = SimpleNamespace(rope_theta=123.0)
        with patch.object(run_retrieval.AutoConfig, "from_pretrained", return_value=existing):
            self.assertEqual(run_retrieval.load_legacy_qwen_config("fixture").rope_theta, 123.0)

        with patch.object(run_retrieval.AutoConfig, "from_pretrained", return_value=SimpleNamespace()):
            with self.assertRaisesRegex(ValueError, "rotary embedding base"):
                run_retrieval.load_legacy_qwen_config("fixture")

    def test_model_loaders_use_the_compatible_config(self):
        for retriever in ("flat-stella", "flat-gte"):
            with self.subTest(retriever=retriever):
                config = SimpleNamespace(rope_parameters={"rope_theta": 1000000.0})
                master = run_retrieval.DenseRetrievalMaster.__new__(run_retrieval.DenseRetrievalMaster)
                master.args = SimpleNamespace(retriever=retriever, cache_dir="/synthetic")
                master.device = torch.device("cpu")
                state = {"linear.weight": torch.zeros(1024, 2), "linear.bias": torch.zeros(1024)}
                with patch.object(run_retrieval.AutoConfig, "from_pretrained", return_value=config), \
                     patch.object(run_retrieval, "load_legacy_qwen_tokenizer", return_value=FakeTokenizer()) as tokenizer_loader, \
                     patch.object(run_retrieval.AutoModel, "from_pretrained", return_value=LoadableModel()) as loader, \
                     patch.object(run_retrieval, "restore_legacy_rotary_buffers") as restore, \
                     patch.object(torch, "load", return_value=state) as load:
                    master.prepare_retriever()
                self.assertIs(loader.call_args.kwargs["config"], config)
                self.assertEqual(tokenizer_loader.call_count, 1)
                self.assertEqual(tokenizer_loader.call_args.args[0], "/synthetic/dunzhang_stella_en_1.5B_v5" if retriever == "flat-stella" else "Alibaba-NLP/gte-Qwen2-7B-instruct")
                self.assertEqual(config.rope_theta, 1000000.0)
                restore.assert_called_once_with(loader.return_value, 513 if retriever == "flat-stella" else 8193)
                if retriever == "flat-stella":
                    self.assertEqual(load.call_args.kwargs, {"map_location": "cpu", "weights_only": True})

    def test_rotary_buffers_are_rebuilt_from_the_model_base(self):
        model = torch.nn.Module()
        model.config = SimpleNamespace(
            rope_theta=1000000.0,
            rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
            num_hidden_layers=1,
        )
        model.weight = torch.nn.Parameter(torch.zeros(1))
        model.rotary = Qwen2RotaryEmbedding()
        with self.assertRaisesRegex(ValueError, "unsupported rotary implementation"):
            run_retrieval.restore_legacy_rotary_buffers(model, 3)
        fixture_hash = hashlib.sha256(inspect.getsource(Qwen2RotaryEmbedding).encode()).hexdigest()
        methods = {
            name: hashlib.sha256(inspect.getsource(getattr(Qwen2RotaryEmbedding, name)).encode()).hexdigest()
            for name in ("_set_cos_sin_cache", "forward")
        }
        with patch.object(run_retrieval, "LEGACY_QWEN_ROTARY_SHA256", fixture_hash), \
             patch.object(run_retrieval, "LEGACY_QWEN_METHOD_SHA256", methods):
            run_retrieval.restore_legacy_rotary_buffers(model, 3)
            original = Qwen2RotaryEmbedding.forward

            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                return original(*args, **kwargs)

            with patch.object(Qwen2RotaryEmbedding, "forward", wrapped):
                with self.assertRaisesRegex(ValueError, "unsupported rotary implementation"):
                    run_retrieval.restore_legacy_rotary_buffers(model, 3)
            model.rotary.forward = lambda length: None
            with self.assertRaisesRegex(ValueError, "unsupported rotary implementation"):
                run_retrieval.restore_legacy_rotary_buffers(model, 3)
            del model.rotary.forward
        expected = 1.0 / (1000000.0 ** (torch.arange(0, 8, 2).float() / 8))
        torch.testing.assert_close(model.rotary.inv_freq, expected)
        self.assertEqual(model.rotary.max_seq_len_cached, 3)
        cos, sin = model.rotary(3)
        torch.testing.assert_close(cos[0], torch.ones(8))
        torch.testing.assert_close(sin[0], torch.zeros(8))
        self.assertEqual(cos.shape, (3, 8))
        cos, _ = model.rotary(5)
        self.assertEqual(cos.shape, (5, 8))

        model.config.num_hidden_layers = 2
        with patch.object(run_retrieval, "LEGACY_QWEN_ROTARY_SHA256", fixture_hash), \
             patch.object(run_retrieval, "LEGACY_QWEN_METHOD_SHA256", methods):
            with self.assertRaisesRegex(ValueError, "layer count"):
                run_retrieval.restore_legacy_rotary_buffers(model, 3)
        model.config.num_hidden_layers = 1
        model.weight.data = model.weight.data.half()
        with self.assertRaisesRegex(ValueError, "dtype"):
            run_retrieval.restore_legacy_rotary_buffers(model, 3)
        model.weight.data = model.weight.data.float()
        model.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.int8), requires_grad=False)
        with self.assertRaisesRegex(ValueError, "dtype"):
            run_retrieval.restore_legacy_rotary_buffers(model, 3)
        model.weight = torch.nn.Parameter(torch.zeros(1))

        model.config.rope_parameters = {"rope_type": "linear", "rope_theta": 1000000.0}
        with self.assertRaisesRegex(ValueError, "unsupported rotary"):
            run_retrieval.restore_legacy_rotary_buffers(model, 3)

    def test_stella_and_gte_retrieval_do_not_use_legacy_cache(self):
        for retriever in ("flat-stella", "flat-gte"):
            with self.subTest(retriever=retriever):
                model = CacheSensitiveModel()
                master = run_retrieval.DenseRetrievalMaster.__new__(run_retrieval.DenseRetrievalMaster)
                if retriever == "flat-stella":
                    master.retriever_model = (FakeTokenizer(), model, torch.nn.Identity())
                else:
                    master.retriever_model = (FakeTokenizer(), model)
                result = master.run_flat_retrieval("red", retriever, ["red", "blue"])
                self.assertEqual(result.tolist(), [0, 1])
                self.assertGreaterEqual(model.calls, 2)


if __name__ == "__main__":
    unittest.main()
