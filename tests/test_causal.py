"""Causal attention captured from ordinary PyTorch (ADR-0020).

Each case checks three things: the capture produces a `CausalMask`, the real Rust checker
(`gg job check-model`) accepts the description, and **the Rust executor computes the same function
as PyTorch** (`gg job eval-model`), which is the only proof that the translation is faithful.
"""

import json
import math
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gpugoblin import UnsupportedOperator, export_model, tensors  # noqa: E402
from gpugoblin.export import constant_parameter_names, state_dict_from_bundle  # noqa: E402
from gg_binary import GG  # noqa: E402

V, T, D, H = 23, 8, 16, 2


def gg(*args, check=True):
    if not GG.exists():
        pytest.skip("gg binary not found (set GG_BIN)")
    out = subprocess.run([str(GG), *args], capture_output=True, text=True)
    if check:
        assert out.returncode == 0, out.stderr
    return out


def evaluate(description, weights, ids, tmp_path):
    """Output of the Rust executor for these weights and inputs."""
    model = tmp_path / "model.json"
    model.write_text(json.dumps(description))
    ck = tmp_path / "ck.ggb"
    tensors.write_bundle(ck, weights)
    inp = tmp_path / "ids.ggt"
    tensors.write_tensor(inp, ids.numpy().astype(np.float32))
    out = tmp_path / "out.ggt"
    gg("job", "eval-model", str(model), "--checkpoint", str(ck),
       "--input", f"{description['inputs'][0]['name']}={inp}", "--out", str(out))
    return tensors.read_tensor(out)


class Block(nn.Module):
    """Embedding, one multi-head attention, projection, language-model head."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.norm = nn.LayerNorm(D)
        self.q, self.k, self.v = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)
        self.proj = nn.Linear(D, D)
        self.head = nn.Linear(D, V)

    def heads(self, x):
        b, t, _ = x.shape
        return x.view(b, t, H, D // H).transpose(1, 2)

    def attend(self, q, k, v):  # pragma: no cover - overridden by the variants
        raise NotImplementedError

    def forward(self, ids):
        b, t = ids.shape
        x = self.emb(ids)
        n = self.norm(x)
        y = self.attend(self.heads(self.q(n)), self.heads(self.k(n)), self.heads(self.v(n)))
        y = y.transpose(1, 2).contiguous().view(b, t, D)
        x = x + self.proj(y)
        return self.head(x).reshape(-1, V)


class NanoGPT(Block):
    """nanoGPT: triangular mask in a buffer, masked_fill with −∞, explicit 1/√d factor."""

    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.tril(torch.ones(T, T)).view(1, 1, T, T))

    def attend(self, q, k, v):
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        return F.softmax(att, dim=-1) @ v


class Triu(Block):
    """Mask built in forward with torch.triu."""

    def attend(self, q, k, v):
        att = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        return torch.softmax(att.masked_fill(mask, float("-inf")), dim=-1) @ v


class Sdpa(Block):
    """scaled_dot_product_attention(is_causal=True), the modern way to write it."""

    def attend(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


class Where(Block):
    """torch.where on a lower-triangular mask, which keeps the past."""

    def attend(self, q, k, v):
        att = (q @ k.transpose(-2, -1)) * (D // H) ** -0.5
        keep = torch.tril(torch.ones(T, T, dtype=torch.bool))
        att = torch.where(keep, att, torch.tensor(float("-inf")))
        return F.softmax(att, dim=-1) @ v


@pytest.mark.parametrize("cls", [NanoGPT, Triu, Sdpa, Where], ids=lambda c: c.__name__)
def test_causal_attention_is_captured_and_computes_the_same_function(cls, tmp_path):
    torch.manual_seed(0)
    model = cls().eval()
    ids = torch.randint(0, V, (3, T))
    desc, weights = export_model(model, ids[:2], "cross_entropy", name=cls.__name__.lower(),
                                 input_dtypes=["i64"], target_shape=[0, T])
    ops = [n["op"] for n in desc["nodes"]]
    assert ops.count("CausalMask") == 1, ops
    # The 1/√d factor is a frozen constant, stated in the description itself.
    consts = [p for p in desc["parameters"] if isinstance(p["init"], dict) and "constant" in p["init"]]
    assert consts and all(p["trainable"] is False for p in consts)
    # Multiplied by 1/√d or divided by √d depending on how it is written: the captured constant is either.
    scale = (D // H) ** -0.5
    values = [p["init"]["constant"]["value"] for p in consts]
    assert any(abs(x - scale) < 1e-6 or abs(x - 1 / scale) < 1e-6 for x in values), values

    report = json.loads(gg("job", "check-model", str(_write(tmp_path, desc)), "--json").stdout)
    assert "CausalMask" in report["ops_used"]

    with torch.no_grad():
        expected = model(ids).numpy()
    got = evaluate(desc, weights, ids, tmp_path)
    assert got.shape == expected.shape
    gap = float(np.abs(got - expected).max())
    assert gap <= 1e-5, f"{cls.__name__}: the Rust executor departs from PyTorch by {gap}"


def test_the_captured_model_does_not_read_the_future(tmp_path):
    """On the captured description: changing the last token changes no earlier position."""
    torch.manual_seed(1)
    model = NanoGPT().eval()
    ids = torch.randint(0, V, (1, T))
    desc, weights = export_model(model, ids, "cross_entropy", input_dtypes=["i64"], target_shape=[0, T])
    base = evaluate(desc, weights, ids, tmp_path)
    changed = ids.clone()
    changed[0, -1] = (changed[0, -1] + 1) % V
    after = evaluate(desc, weights, changed, tmp_path)
    np.testing.assert_array_equal(base[: T - 1], after[: T - 1])
    assert not np.array_equal(base[T - 1], after[T - 1])


def test_a_mask_that_is_not_causal_is_rejected_by_name():
    class Banded(Block):
        def attend(self, q, k, v):
            att = q @ k.transpose(-2, -1)
            # A band: each position sees only the two previous ones. Not the causal mask.
            band = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1) | torch.tril(
                torch.ones(T, T, dtype=torch.bool), diagonal=-3)
            return F.softmax(att.masked_fill(band, float("-inf")), dim=-1) @ v

    with pytest.raises(UnsupportedOperator) as e:
        export_model(Banded(), torch.randint(0, V, (2, T)), "cross_entropy", input_dtypes=["i64"],
                     target_shape=[0, T])
    assert "triangle" in str(e.value)


def test_a_zero_fill_is_not_a_causal_mask():
    class ZeroFill(Block):
        def attend(self, q, k, v):
            att = q @ k.transpose(-2, -1)
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
            return F.softmax(att.masked_fill(mask, 0.0), dim=-1) @ v

    with pytest.raises(UnsupportedOperator) as e:
        export_model(ZeroFill(), torch.randint(0, V, (2, T)), "cross_entropy", input_dtypes=["i64"],
                     target_shape=[0, T])
    assert "−∞" in str(e.value)


def test_checkpoint_round_trip_skips_synthetic_constants():
    torch.manual_seed(2)
    model = NanoGPT().eval()
    desc, weights = export_model(model, torch.randint(0, V, (2, T)), "cross_entropy",
                                 input_dtypes=["i64"], target_shape=[0, T])
    names = constant_parameter_names(desc)
    assert names, "the 1/√d factor must appear as a constant"
    fresh = NanoGPT()
    state_dict_from_bundle(fresh, tensors.decode_bundle(tensors.encode_bundle(weights)), ignore=names)
    for (ka, a), (kb, b2) in zip(model.state_dict().items(), fresh.state_dict().items()):
        assert torch.equal(a, b2), ka


def _write(tmp_path, desc):
    p = tmp_path / "model.json"
    p.write_text(json.dumps(desc))
    return p
