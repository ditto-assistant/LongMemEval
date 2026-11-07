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


class RetrievalCompatibilityTest(unittest.TestCase):
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
                     patch.object(run_retrieval.AutoTokenizer, "from_pretrained", return_value=FakeTokenizer()), \
                     patch.object(run_retrieval.AutoModel, "from_pretrained", return_value=LoadableModel()) as loader, \
                     patch.object(torch, "load", return_value=state) as load:
                    master.prepare_retriever()
                self.assertIs(loader.call_args.kwargs["config"], config)
                self.assertEqual(config.rope_theta, 1000000.0)
                if retriever == "flat-stella":
                    self.assertEqual(load.call_args.kwargs, {"map_location": "cpu", "weights_only": True})

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
