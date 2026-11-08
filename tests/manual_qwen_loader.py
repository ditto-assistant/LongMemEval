"""Offline CPU probe for the reviewed public Qwen model implementations.

Run in a network-disabled, unprivileged container with a read-only model
directory containing the four exact files identified below. This deliberately
executes model Python code; it is not part of automatic CI.

`gte.py` and `gte-config.json` are modeling_qwen.py and config.json from
Alibaba-NLP/gte-Qwen2-7B-instruct revision a8d08b36ada9cacfe34c4d6f80957772a025daf2.
`stella.py` and `stella-config.json` are the same files from
NovaSearch/stella_en_1.5B_v5 revision 7817065102fd9e1b031fe874e910c01f40b2f001.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
import transformers
from safetensors.torch import save_file

from src.retrieval import run_retrieval


EXPECTED = {
    "gte.py": "99ba535c53fb52c61067ddbc7f70e0b877ec3a2aa53bd89a559432a3efc7f0a7",
    "gte-config.json": "9e8b024c76885278a9359310925cd78549792d3853082b413c1b21b5be72716f",
    "stella.py": "8b56872fee78d974adadfd048eaa5ea2798e0ddaab92c24f710613f5a1f7ea9d",
    "stella-config.json": "0939230fc285587deee40b1484bb0f80144797db1bfe84becf4e41d82f4e91f5",
}


def check_model(name, inputs):
    with tempfile.TemporaryDirectory(prefix="hf-local-") as directory:
        directory = Path(directory)
        raw = json.loads((inputs / f"{name}-config.json").read_text())
        raw.update(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )
        (directory / "config.json").write_text(json.dumps(raw))
        shutil.copyfile(inputs / f"{name}.py", directory / "modeling_qwen.py")
        config = run_retrieval.load_legacy_qwen_config(str(directory))
        assert config.rope_theta == 1000000.0
        config._attn_implementation = "eager"
        spec = importlib.util.spec_from_file_location(f"public_{name}", directory / "modeling_qwen.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reference = module.Qwen2Model(config).eval()
        with torch.no_grad():
            for parameter in reference.parameters():
                values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
                parameter.copy_((values.remainder(19) - 9) * 0.002)
        save_file({k: v.contiguous() for k, v in reference.state_dict().items()}, str(directory / "model.safetensors"))
        config = run_retrieval.load_legacy_qwen_config(str(directory))
        candidate = transformers.AutoModel.from_pretrained(
            str(directory), config=config, trust_remote_code=True, local_files_only=True
        ).eval()
        assert candidate.config.rope_theta == 1000000.0
        ids = torch.tensor([[1, 2, 3], [3, 2, 0]])
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        with torch.no_grad():
            expected = reference(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            for component in candidate.modules():
                if type(component).__name__ == "Qwen2RotaryEmbedding":
                    for key in ("inv_freq", "cos_cached", "sin_cached"):
                        getattr(component, key).fill_(float("nan"))
            poisoned = candidate(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            assert not torch.isfinite(poisoned).all(), "poisoned-buffer control did not fail"
            run_retrieval.restore_legacy_rotary_buffers(candidate, 5)
            actual = candidate(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        active = mask.bool().unsqueeze(-1).expand_as(expected)
        assert torch.isfinite(expected[active]).all() and torch.isfinite(actual[active]).all()
        delta = (expected[active] - actual[active]).abs().max().item()
        assert torch.allclose(expected[active], actual[active], rtol=1e-5, atol=1e-6), delta
        return {"model": name, "attention": candidate.config._attn_implementation, "max_abs_diff_active": delta}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("Set HF_HUB_OFFLINE=1 and run in a network-disabled container")
    for filename, expected in EXPECTED.items():
        actual = hashlib.sha256((args.model_dir / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Unexpected source or config hash: {filename}")
    torch.set_num_threads(1)
    for name in ("gte", "stella"):
        print(json.dumps(check_model(name, args.model_dir)))


if __name__ == "__main__":
    main()
