"""Position encoding captured from ordinary PyTorch (ADR-0021).

Three ways of writing it: learned positions as in GPT-2 / nanoGPT, the original sinusoidal table in
a buffer, and Hugging Face-style RoPE rotations. For each one, the Rust executor must compute
**the same function** as PyTorch (`gg job eval-model`).
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
from gpugoblin.export import (  # noqa: E402
    constant_parameter_names,
    position_table,
    recognise_position_table,
    state_dict_from_bundle,
)
from gg_binary import GG  # noqa: E402

V, T, D, H = 19, 8, 16, 2


def gg(*args):
    if not GG.exists():
        pytest.skip("gg binary not found (set GG_BIN)")
    out = subprocess.run([str(GG), *args], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out


def evaluate(description, weights, ids, tmp_path, checkpoint=True):
    model = tmp_path / "model.json"
    model.write_text(json.dumps(description))
    inp = tmp_path / "ids.ggt"
    tensors.write_tensor(inp, ids.numpy().astype(np.float32))
    out = tmp_path / "out.ggt"
    name = description["inputs"][0]["name"]
    args = ["job", "eval-model", str(model), "--input", f"{name}={inp}", "--out", str(out)]
    if checkpoint:
        ck = tmp_path / "ck.ggb"
        tensors.write_bundle(ck, weights)
        args += ["--checkpoint", str(ck)]
    gg(*args)
    return tensors.read_tensor(out)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Lm(nn.Module):
    """Small causal language model; subclasses choose the position encoding."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.norm = nn.LayerNorm(D)
        self.q, self.k, self.v = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)
        self.proj = nn.Linear(D, D)
        self.head = nn.Linear(D, V)

    def positions(self, x):
        return x

    def rotate(self, q, k):
        return q, k

    def forward(self, ids):
        b, t = ids.shape
        x = self.positions(self.emb(ids))
        n = self.norm(x)
        split = lambda z: z.view(b, t, H, D // H).transpose(1, 2)  # noqa: E731
        q, k = self.rotate(split(self.q(n)), split(self.k(n)))
        y = F.scaled_dot_product_attention(q, k, split(self.v(n)), is_causal=True)
        x = x + self.proj(y.transpose(1, 2).contiguous().view(b, t, D))
        return self.head(x).reshape(-1, V)


class Learned(Lm):
    """GPT-2 / nanoGPT: `wpe(arange(t))`, with block_size larger than the sequence."""

    def __init__(self, block_size=T):
        super().__init__()
        self.wpe = nn.Embedding(block_size, D)

    def positions(self, x):
        return x + self.wpe(torch.arange(0, x.shape[1], dtype=torch.long))


class Sinusoidal(Lm):
    """The original paper's table, in a buffer, sliced to the sequence length."""

    def __init__(self, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, D)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def positions(self, x):
        return x + self.pe[:, : x.shape[1]]


class Rope(Lm):
    """Hugging Face-style RoPE: inv_freq in a non-persistent buffer, cos and sin computed in
    forward, half rotation `rotate_half`."""

    def __init__(self, base=10000.0):
        super().__init__()
        hd = D // H
        inv_freq = 1.0 / (base ** (torch.arange(0, hd, 2, dtype=torch.int64).float() / hd))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotate(self, q, k):
        t = torch.arange(q.shape[-2], dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos()[None, None], emb.sin()[None, None]
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


CASES = [
    ("learned", lambda: Learned(), None),
    ("learned-sliced", lambda: Learned(block_size=32), None),
    ("sinusoidal", lambda: Sinusoidal(), "sinusoidal"),
    ("rope-10k", lambda: Rope(10000.0), "rotary_cos"),
    ("rope-500k", lambda: Rope(500000.0), "rotary_cos"),
]


@pytest.mark.parametrize("label,make,table", CASES, ids=[c[0] for c in CASES])
def test_positions_are_captured_and_compute_the_same_function(label, make, table, tmp_path):
    torch.manual_seed(0)
    model = make().eval()
    ids = torch.randint(0, V, (3, T))
    desc, weights = export_model(model, ids[:2], "cross_entropy", name=label,
                                 input_dtypes=["i64"], target_shape=[0, T])
    inits = [p["init"] for p in desc["parameters"] if isinstance(p["init"], dict)]
    if table:
        found = [i[table]["base"] for i in inits if table in i]
        assert found, f"{label}: table {table} not recognized, inits {inits}"
        if label.startswith("rope"):
            expected = 500000.0 if label.endswith("500k") else 10000.0
            assert all(abs(b - expected) < 1e-6 * expected for b in found), found
            assert any("rotary_sin" in i for i in inits)
    if label == "learned-sliced":
        assert any(n["op"] == "Slice" for n in desc["nodes"]), "expected a sliced wpe"
    report = json.loads(gg("job", "check-model", str(_write(tmp_path, desc)), "--json").stdout)
    assert report["trainable_parameters"] == sum(p.numel() for p in model.parameters())

    with torch.no_grad():
        expected = model(ids).numpy()
    got = evaluate(desc, weights, ids, tmp_path)
    gap = float(np.abs(got - expected).max())
    assert gap <= 1e-5, f"{label}: the Rust executor departs from PyTorch by {gap}"
    if table:
        # Without a checkpoint, the tables come from the Rust generator named in the description:
        # the description is self-sufficient, up to f32 rounding.
        auto = evaluate(desc, {}, ids, tmp_path, checkpoint=False)
        assert auto.shape == expected.shape


def test_position_tables_are_recognized_by_value_and_nothing_else_is():
    for kind in ("rotary_cos", "rotary_sin", "sinusoidal"):
        for base in (10000.0, 500000.0, 1000000.0):
            for t_len, d in ((16, 8), (64, 32), (512, 64)):
                tab = position_table(kind, base, t_len, d)
                got = recognise_position_table(tab[None, None])
                assert got is not None and kind in got, (kind, base, t_len, d, got)
                assert abs(got[kind]["base"] - base) <= 1e-6 * base
    # Tables that are close but wrong do not pass: another base, noise, another shape.
    assert recognise_position_table(position_table("rotary_cos", 10000.0, 32, 16) + 1e-2) is None
    assert recognise_position_table(torch.randn(1, 32, 16)) is None
    assert recognise_position_table(position_table("rotary_cos", 10000.0, 32, 16)[None].repeat(2, 1, 1)) is None


def test_an_arbitrary_constant_is_rejected_by_name():
    class Noise(Lm):
        def __init__(self):
            super().__init__()
            self.register_buffer("noise", torch.randn(1, T, D))

        def positions(self, x):
            return x + self.noise

    with pytest.raises(UnsupportedOperator) as e:
        export_model(Noise(), torch.randint(0, V, (2, T)), "cross_entropy", input_dtypes=["i64"],
                     target_shape=[0, T])
    assert "non-uniform constant" in str(e.value)


def test_trained_weights_reload_into_the_module():
    torch.manual_seed(3)
    model = Rope().eval()
    desc, weights = export_model(model, torch.randint(0, V, (2, T)), "cross_entropy",
                                 input_dtypes=["i64"], target_shape=[0, T])
    names = constant_parameter_names(desc)
    assert any(n.startswith("pos.") for n in names), names
    fresh = Rope()
    state_dict_from_bundle(fresh, tensors.decode_bundle(tensors.encode_bundle(weights)), ignore=names)
    for (name, a), (_, b) in zip(model.named_parameters(), fresh.named_parameters()):
        assert torch.equal(a, b), name


def _write(tmp_path, desc):
    p = tmp_path / "model.json"
    p.write_text(json.dumps(desc))
    return p


@pytest.mark.parametrize("kind", ["rotary_cos", "rotary_sin", "sinusoidal"])
def test_rust_generator_matches_the_reference(kind, tmp_path):
    """The Rust generator, not the Python reference, compared with the reference table: a graph
    whose output is the table itself, evaluated by `gg job eval-model` without a checkpoint."""
    t_len, d, base = 64, 32, 10000.0
    desc = {
        "format": "gg-graph/v1", "name": "table",
        "inputs": [{"name": "x", "shape": [0, t_len, d]}],
        "parameters": [
            {"name": "w", "shape": [1], "init": "ones"},
            {"name": "tab", "shape": [1, t_len, d], "init": {kind: {"base": base}}, "trainable": False},
        ],
        "nodes": [
            {"op": "Mul", "inputs": ["x", "w"], "output": "zero"},
            {"op": "Add", "inputs": ["zero", "tab"], "output": "out"},
        ],
        "loss": {"mean_squared_error": {"prediction": "out", "target": {"name": "y", "shape": [0, t_len, d]}}},
    }
    model = tmp_path / "m.json"
    model.write_text(json.dumps(desc))
    inp = tmp_path / "x.ggt"
    tensors.write_tensor(inp, np.zeros((1, t_len, d), dtype=np.float32))
    out = tmp_path / "o.ggt"
    gg("job", "eval-model", str(model), "--input", f"x={inp}", "--out", str(out))
    rust = tensors.read_tensor(out).reshape(t_len, d)
    reference = position_table(kind, base, t_len, d).numpy()
    gap = float(np.abs(rust - reference).max())
    # Same formula in f32; only sin, cos and pow may differ by one unit in the last place.
    assert gap <= 5e-6, f"{kind}: the Rust generator departs from the reference by {gap}"
