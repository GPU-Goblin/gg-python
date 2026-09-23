"""Real `transformers` models captured as they are (gg-server/docs/ROADMAP.md, P11 C3).

Llama (full heads and grouped queries), Qwen2 (bias on q, k, v), Mistral: SwiGLU, hand-written
RMSNorm (`pow`, `mean`, `rsqrt`), key-head repetition, RoPE. For each one, the Rust executor must
compute **the same function** as PyTorch, and a frozen module (`requires_grad=False`) must stay
frozen in the description.

`transformers` is not a dependency of the package: without it, these tests are skipped, which is
not a success.
"""

import json
import logging
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from torch import nn  # noqa: E402

from gpugoblin import export_model, tensors  # noqa: E402
from gg_binary import GG  # noqa: E402

logging.getLogger("transformers").setLevel(logging.ERROR)
V, T = 64, 8


class Logits(nn.Module):
    """The language model, returned in the shape the loss expects: `[batch·T, V]`."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, ids):
        return self.lm(input_ids=ids, use_cache=False).logits.reshape(-1, V)


def build(kind, kv_heads):
    common = dict(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=kv_heads, max_position_embeddings=32)
    if kind == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM
        return LlamaForCausalLM(LlamaConfig(**common))
    if kind == "qwen2":
        from transformers import Qwen2Config, Qwen2ForCausalLM
        return Qwen2ForCausalLM(Qwen2Config(**common))
    if kind == "mistral":
        from transformers import MistralConfig, MistralForCausalLM
        return MistralForCausalLM(MistralConfig(sliding_window=None, **common))
    if kind == "llama3.1":
        from transformers import LlamaConfig, LlamaForCausalLM
        rope = {"rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0, "low_freq_factor": 1.0,
                "high_freq_factor": 4.0, "original_max_position_embeddings": 4}
        return LlamaForCausalLM(LlamaConfig(rope_parameters=rope, **common))
    raise ValueError(kind)


def gg(*args):
    if not GG.exists():
        pytest.skip("gg binary not found (set GG_BIN)")
    out = subprocess.run([str(GG), *args], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out


CASES = [("llama", 4), ("llama", 2), ("llama", 1), ("qwen2", 2), ("mistral", 2), ("llama3.1", 2)]


@pytest.mark.parametrize("kind,kv", CASES, ids=[f"{k}-kv{v}" for k, v in CASES])
def test_a_real_transformers_model_computes_the_same_function(kind, kv, tmp_path):
    torch.manual_seed(0)
    model = Logits(build(kind, kv)).eval()
    # Nonzero weights everywhere, otherwise a translation error could hide behind a zero.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02 * torch.randn_like(p))
    ids = torch.randint(0, V, (3, T))
    desc, weights = export_model(model, ids[:2], "cross_entropy", name=f"{kind}-kv{kv}",
                                 input_dtypes=["i64"], target_shape=[0, T])
    ops = {n["op"] for n in desc["nodes"]}
    assert {"ReduceMean", "Rsqrt", "Sigmoid", "CausalMask"} <= ops, ops
    if kv < 4:
        assert "Concat" in ops, "expected key-head repetition"
    path = tmp_path / "m.json"
    path.write_text(json.dumps(desc))
    report = json.loads(gg("job", "check-model", str(path), "--json").stdout)
    assert report["trainable_parameters"] == sum(p.numel() for p in model.parameters())

    ck = tmp_path / "ck.ggb"
    tensors.write_bundle(ck, weights)
    inp = tmp_path / "ids.ggt"
    tensors.write_tensor(inp, ids.numpy().astype(np.float32))
    out = tmp_path / "out.ggt"
    gg("job", "eval-model", str(path), "--checkpoint", str(ck), "--input", f"x={inp}", "--out", str(out))
    got = tensors.read_tensor(out)
    with torch.no_grad():
        expected = model(ids).numpy()
    gap = float(np.abs(got - expected).max())
    print(f"{kind} kv={kv}: {len(desc['nodes'])} nodes, gap {gap:.1e}")
    assert gap <= 1e-5, f"{kind} kv={kv}: the Rust executor departs from PyTorch by {gap}"


def test_a_frozen_base_stays_frozen(tmp_path):
    """Frozen base, only the head trained: the description says so, and the count follows."""
    torch.manual_seed(1)
    model = Logits(build("llama", 2)).eval()
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("lm.lm_head"))
    desc, _ = export_model(model, torch.randint(0, V, (2, T)), "cross_entropy",
                           input_dtypes=["i64"], target_shape=[0, T])
    trainable = [p["name"] for p in desc["parameters"] if p.get("trainable", True)]
    assert trainable == ["lm.lm_head.weight"], trainable
    path = tmp_path / "m.json"
    path.write_text(json.dumps(desc))
    report = json.loads(gg("job", "check-model", str(path), "--json").stdout)
    assert report["trainable_parameters"] == model.lm.lm_head.weight.numel()
