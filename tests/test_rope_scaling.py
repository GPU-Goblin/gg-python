"""Scaled RoPE (ADR-0022), compared with `transformers` itself.

Three paths, three proofs:

1. **The Rust skeleton** (`gg job template --rope-scaling`) reproduces the frequencies and the
   amplitude that `transformers` computes for each variant (linear, dynamic NTK, YaRN, LongRoPE,
   Llama 3.1);
2. **the PyTorch driver** recovers, from the values of a pair of tables alone, the module's exact
   f32 frequencies, whatever the variant that produced them;
3. **the real `LlamaRotaryEmbedding` class** from `transformers`, once captured, computes the same
   function in the Rust executor as in PyTorch.

`transformers` is not a dependency of the package: without it, these tests are skipped, which is
not a success.
"""

import json
import logging
import math
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import LlamaConfig  # noqa: E402
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS  # noqa: E402
from transformers.models.llama.modeling_llama import (  # noqa: E402
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

from gpugoblin import ExportError, export_model, freeze_rotary_embeddings, tensors  # noqa: E402
from gpugoblin.export import (  # noqa: E402
    recognise_position_table,
    recognise_rotary_pair,
    recover_rotary,
    rotary_table,
)
from gg_binary import GG  # noqa: E402

# transformers warns loudly about unusual configurations; the tests use some.
logging.getLogger("transformers").setLevel(logging.ERROR)

HD = 64
SHORT = [1.0 + 0.01 * i for i in range(HD // 2)]
LONG = [1.0 + 0.5 * i for i in range(HD // 2)]

#: (name, Hugging Face parameters, max_position_embeddings, sequence length)
VARIANTS = [
    ("default", {"rope_type": "default", "rope_theta": 500000.0}, 8192, 64),
    ("linear", {"rope_type": "linear", "rope_theta": 10000.0, "factor": 4.0}, 16384, 64),
    ("linear-2.7", {"rope_type": "linear", "rope_theta": 10000.0, "factor": 2.7}, 16384, 64),
    ("dynamic-inside", {"rope_type": "dynamic", "rope_theta": 10000.0, "factor": 2.0}, 2048, 1024),
    ("dynamic-beyond", {"rope_type": "dynamic", "rope_theta": 10000.0, "factor": 2.0}, 2048, 3000),
    ("dynamic-odd", {"rope_type": "dynamic", "rope_theta": 1000000.0, "factor": 3.3}, 1000, 7777),
    ("llama3.1", {"rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0,
                  "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                  "original_max_position_embeddings": 8192}, 131072, 64),
    ("llama3.2", {"rope_type": "llama3", "rope_theta": 500000.0, "factor": 32.0,
                  "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                  "original_max_position_embeddings": 8192}, 131072, 64),
    ("yarn-qwen", {"rope_type": "yarn", "rope_theta": 1000000.0, "factor": 4.0,
                   "original_max_position_embeddings": 32768}, 131072, 64),
    ("yarn-deepseek", {"rope_type": "yarn", "rope_theta": 10000.0, "factor": 40.0,
                       "original_max_position_embeddings": 4096, "beta_fast": 32, "beta_slow": 1,
                       "mscale": 0.707, "mscale_all_dim": 0.707}, 163840, 64),
    ("yarn-mscale", {"rope_type": "yarn", "rope_theta": 10000.0, "factor": 40.0,
                     "original_max_position_embeddings": 4096, "mscale": 1.0,
                     "mscale_all_dim": 0.5}, 163840, 64),
    ("yarn-untruncated", {"rope_type": "yarn", "rope_theta": 150000.0, "factor": 32.0,
                          "original_max_position_embeddings": 4096, "truncate": False}, 131072, 64),
    ("yarn-attention", {"rope_type": "yarn", "rope_theta": 10000.0, "factor": 8.0,
                        "original_max_position_embeddings": 4096, "attention_factor": 1.25},
     32768, 64),
    ("longrope-short", {"rope_type": "longrope", "rope_theta": 10000.0, "short_factor": SHORT,
                        "long_factor": LONG, "original_max_position_embeddings": 4096}, 131072, 64),
    ("longrope-long", {"rope_type": "longrope", "rope_theta": 10000.0, "short_factor": SHORT,
                       "long_factor": LONG, "original_max_position_embeddings": 4096}, 131072, 5000),
    ("legacy-type", {"type": "linear", "rope_theta": 10000.0, "factor": 2.0}, 8192, 64),
]


def gg(*args):
    if not GG.exists():
        pytest.skip("gg binary not found (set GG_BIN)")
    out = subprocess.run([str(GG), *args], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out


def reference(rope_parameters, max_positions, seq, head_dim=HD):
    """Frequencies and amplitude from `transformers` at length `seq`, as in a forward pass."""
    config = LlamaConfig(hidden_size=head_dim * 2, num_attention_heads=2, head_dim=head_dim,
                         max_position_embeddings=max_positions,
                         rope_parameters=dict(rope_parameters))
    kind = config.rope_parameters["rope_type"]
    if kind == "default":
        return LlamaRotaryEmbedding.compute_default_rope_parameters(config)
    if kind == "dynamic" and seq > max_positions:
        # In a forward pass, the length arrives as a tensor: the base is recomputed in f32.
        return ROPE_INIT_FUNCTIONS[kind](config, None, seq_len=torch.tensor(seq))
    if kind == "longrope":
        return ROPE_INIT_FUNCTIONS[kind](config, None, seq_len=seq)
    return ROPE_INIT_FUNCTIONS[kind](config, None)


def reference_tables(inv_freq, amplitude, t_len):
    """Cos and sin tables `[T, d]` computed as `LlamaRotaryEmbedding.forward` does."""
    position = torch.arange(t_len)[None].float()
    freqs = (inv_freq[None, :, None].float() @ position[:, None, :]).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return (emb.cos() * amplitude)[0], (emb.sin() * amplitude)[0]


def ulps(a, b):
    return int((a.view(torch.int32).long() - b.view(torch.int32).long()).abs().max())


def law_frequencies(law, d):
    if "base" in law:
        base = torch.tensor(law["base"], dtype=torch.float32)
        return 1.0 / (base ** (torch.arange(0, d, 2, dtype=torch.int64).float() / d)), 1.0
    freqs = torch.tensor(law["frequencies"], dtype=torch.float64).float()
    return freqs, law.get("amplitude", 1.0)


@pytest.mark.parametrize("label,params,max_positions,seq", VARIANTS, ids=[v[0] for v in VARIANTS])
def test_rust_template_matches_transformers(label, params, max_positions, seq, tmp_path):
    """The Rust formulas of `gg job template --rope-scaling`, against the reference ones.

    Both compute in f32 with the same order of operations. What remains is the f32 power: `torch`
    evaluates it with SLEEF, Rust with the system libm, and the two sometimes differ by one unit in
    the last place. Measured on 2026-09-21: 14 variants out of 16 byte-identical, the other two
    within 1 and 2 ulp (YaRN without truncation, dynamic NTK beyond the context).
    """
    inv_freq, attention = reference(params, max_positions, seq)
    out = tmp_path / "t.json"
    gg("job", "template", "transformer", "--layers", "1", "--dim", str(HD * 2), "--heads", "2",
       "--seq", str(seq), "--vocab", "32", "--rope-scaling", json.dumps(params),
       "--rope-max-positions", str(max_positions), "--out", str(out))
    description = json.loads(out.read_text())
    law = next(p["init"]["rotary_cos"] for p in description["parameters"] if p["name"] == "rope.cos")
    ours, amplitude = law_frequencies(law, HD)
    gap = ulps(ours, inv_freq.float())
    print(f"{label}: {'base' if 'base' in law else 'explicit'}, "
          f"{int((ours == inv_freq).sum())}/{HD // 2} identical frequencies, max gap {gap} ulp")
    assert gap <= 2, f"{label}: {gap} ulp"
    assert np.float32(amplitude) == np.float32(attention), (amplitude, attention)


def test_rust_template_rejects_what_it_does_not_know(tmp_path):
    if not GG.exists():
        pytest.skip("gg binary not found (set GG_BIN)")
    for params, needle in (
        ({"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096,
          "mrope_section": [16, 24, 24]}, "mrope_section"),
        ({"rope_type": "proportional", "factor": 1.0}, "variant 'proportional'"),
        ({"rope_type": "default", "partial_rotary_factor": 0.25}, "partial rotation"),
    ):
        out = subprocess.run(
            [str(GG), "job", "template", "transformer", "--layers", "1", "--dim", "64",
             "--heads", "2", "--seq", "16", "--vocab", "8", "--rope-scaling", json.dumps(params),
             "--out", str(tmp_path / "t.json")],
            capture_output=True, text=True)
        assert out.returncode != 0 and needle in out.stderr, out.stderr
    # A base passed separately cannot contradict the one in the dictionary.
    out = subprocess.run(
        [str(GG), "job", "template", "transformer", "--layers", "1", "--dim", "64", "--heads", "2",
         "--seq", "16", "--vocab", "8", "--rope-base", "10000",
         "--rope-scaling", json.dumps({"rope_type": "default", "rope_theta": 500000.0}),
         "--out", str(tmp_path / "t.json")], capture_output=True, text=True)
    assert out.returncode != 0 and "contradicts" in out.stderr, out.stderr


@pytest.mark.parametrize("label,params,max_positions,seq", VARIANTS, ids=[v[0] for v in VARIANTS])
@pytest.mark.parametrize("t_len", [16, 512])
def test_a_table_pair_gives_back_the_exact_frequencies(label, params, max_positions, seq, t_len):
    """From the values of the two tables alone, the driver recovers the module's exact f32
    frequencies and its amplitude: this is what makes recognition independent of the variant."""
    inv_freq, attention = reference(params, max_positions, seq)
    cos, sin = reference_tables(inv_freq, attention, t_len)
    frequencies, amplitude = recover_rotary(cos, sin)
    got = torch.tensor(frequencies, dtype=torch.float64).float()
    regenerated = [rotary_table(k, frequencies, amplitude, t_len) for k in ("rotary_cos", "rotary_sin")]
    assert amplitude == float(np.float32(attention))
    # A frequency that does not rotate enough over t_len positions to stand out from its
    # neighbors cannot be identified; the table itself must be reproduced byte for byte.
    assert torch.equal(regenerated[0], cos) and torch.equal(regenerated[1], sin), label
    if t_len >= 512:
        assert torch.equal(got, inv_freq.float()), f"{label}: {ulps(got, inv_freq.float())} ulp"


def test_the_pair_names_the_law_it_can():
    inv_freq, _ = reference({"rope_type": "default", "rope_theta": 500000.0}, 8192, 64)
    assert recognise_rotary_pair(*reference_tables(inv_freq, 1.0, 64)) == {"base": 500000.0}
    inv_freq, attention = reference(VARIANTS[8][1], 131072, 64)  # YaRN
    law = recognise_rotary_pair(*reference_tables(inv_freq, attention, 64))
    assert set(law) == {"frequencies", "amplitude"} and law["amplitude"] > 1.1


def test_impostors_are_rejected():
    inv_freq, attention = reference(VARIANTS[6][1], 131072, 64)  # Llama 3.1
    cos, sin = reference_tables(inv_freq, attention, 64)
    other, _ = reference({"rope_type": "default", "rope_theta": 10000.0}, 8192, 64)
    _, sin_other = reference_tables(other, 1.0, 64)
    for label, c, s in (
        ("negated sine", cos, -sin),
        ("sine of another law", cos, sin_other),
        ("noisy cosine", cos + 1e-3 * torch.randn_like(cos), sin),
        ("different amplitudes", cos, sin * 1.01),
        ("cosine shifted by one position", torch.roll(cos, 1, 0), sin),
        ("noise", torch.randn(64, HD), torch.randn(64, HD)),
    ):
        assert recognise_rotary_pair(c, s) is None, label
    # On its own, a scaled table is not an original law: it is not recognized.
    assert recognise_position_table(sin[None, None]) is None


class Lm(nn.Module):
    """Small causal model that uses the real rotary class from `transformers`."""

    V, T, D, H = 19, 8, 32, 2

    def __init__(self, rope_parameters, max_positions=64, head_dim=None, partial=None):
        super().__init__()
        hd = self.D // self.H
        extra = {"partial_rotary_factor": partial} if partial else {}
        config = LlamaConfig(hidden_size=self.D, num_attention_heads=self.H, head_dim=hd,
                             max_position_embeddings=max_positions,
                             rope_parameters={**rope_parameters, **extra})
        self.rotary_dim = int(hd * (partial or 1.0))
        self.rope = LlamaRotaryEmbedding(config)
        self.emb = nn.Embedding(self.V, self.D)
        self.q, self.k, self.v = (nn.Linear(self.D, self.D) for _ in range(3))
        self.head = nn.Linear(self.D, self.V)

    def forward(self, ids):
        b, t = ids.shape
        x = self.emb(ids)
        split = lambda z: z.view(b, t, self.H, self.D // self.H).transpose(1, 2)  # noqa: E731
        q, k = split(self.q(x)), split(self.k(x))
        cos, sin = self.rope(x, torch.arange(t)[None])
        r = self.rotary_dim
        if r == q.shape[-1]:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        else:
            # Phi / GPT-NeoX-style partial rotation: only part of the head rotates.
            q_rot, k_rot = apply_rotary_pos_emb(q[..., :r], k[..., :r], cos, sin)
            q, k = torch.cat((q_rot, q[..., r:]), dim=-1), torch.cat((k_rot, k[..., r:]), dim=-1)
        y = F.scaled_dot_product_attention(q, k, split(self.v(x)), is_causal=True)
        return self.head(x + y.transpose(1, 2).reshape(b, t, self.D)).reshape(-1, self.V)


SMALL = [
    ("default", {"rope_type": "default", "rope_theta": 10000.0}, 64, None, "base"),
    ("linear", {"rope_type": "linear", "rope_theta": 10000.0, "factor": 4.0}, 64, None, "explicit"),
    ("llama3", {"rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0, "low_freq_factor": 1.0,
                "high_freq_factor": 4.0, "original_max_position_embeddings": 4}, 64, None, "explicit"),
    ("yarn", {"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0,
              "original_max_position_embeddings": 4}, 64, None, "explicit"),
    ("dynamic", {"rope_type": "dynamic", "rope_theta": 10000.0, "factor": 2.0}, 4, None, "base"),
    ("longrope", {"rope_type": "longrope", "rope_theta": 10000.0,
                  "short_factor": [1.0 + 0.1 * i for i in range(8)],
                  "long_factor": [1.0 + 0.5 * i for i in range(8)],
                  "original_max_position_embeddings": 4}, 64, None, "explicit"),
    ("partial-yarn", {"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0,
                      "original_max_position_embeddings": 4}, 64, 0.5, "explicit"),
]


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


def rust_parameter(description, name, ids, tmp_path):
    """Value of a parameter as the Rust executor initializes it from the description."""
    model = tmp_path / "gen.json"
    model.write_text(json.dumps(description))
    inp = tmp_path / "gen-ids.ggt"
    tensors.write_tensor(inp, ids.numpy().astype(np.float32))
    out = tmp_path / "gen.ggt"
    gg("job", "eval-model", str(model), "--input", f"{description['inputs'][0]['name']}={inp}",
       "--output", name, "--out", str(out))
    return tensors.read_tensor(out)


@pytest.mark.parametrize("label,params,max_positions,partial,form", SMALL, ids=[s[0] for s in SMALL])
def test_the_real_transformers_rotary_class_is_captured(label, params, max_positions, partial, form,
                                                       tmp_path):
    torch.manual_seed(0)
    model = Lm(params, max_positions, partial=partial).eval()
    ids = torch.randint(0, Lm.V, (3, Lm.T))
    if params["rope_type"] in ("dynamic", "longrope"):
        # These variants choose their frequencies from the length, inside the forward pass:
        # torch.export rejects that, and says so, until the module is frozen.
        with pytest.raises(ExportError, match="freeze_rotary_embeddings"):
            export_model(model, ids[:2], "cross_entropy", input_dtypes=["i64"],
                         target_shape=[0, Lm.T])
        with torch.no_grad():
            before = model(ids)
        assert freeze_rotary_embeddings(model, Lm.T) == ["rope"]
        with torch.no_grad():
            assert torch.equal(model(ids), before), "freezing changed the function"
    description, weights = export_model(model, ids[:2], "cross_entropy", name=label,
                                        input_dtypes=["i64"], target_shape=[0, Lm.T])
    laws = [p["init"] for p in description["parameters"]
            if isinstance(p["init"], dict) and ("rotary_cos" in p["init"] or "rotary_sin" in p["init"])]
    assert len(laws) == 2, laws
    cos_law, sin_law = (next(iter(law.values())) for law in laws)
    # Both tables of a pair carry the same law: their descriptions do not diverge.
    assert cos_law == sin_law
    assert ("base" in cos_law) == (form == "base"), cos_law
    width = [p["shape"][-1] for p in description["parameters"] if "rotary_cos" in p["init"]][0]
    assert width == model.rotary_dim

    with torch.no_grad():
        expected = model(ids).numpy()
    got = evaluate(description, weights, ids, tmp_path)
    gap = float(np.abs(got - expected).max())
    assert gap <= 1e-5, f"{label}: the Rust executor departs from PyTorch by {gap}"
    # The tables as the Rust generator produces them from the description, with the module's
    # weights: the description is self-sufficient.
    generated = dict(weights)
    for name in (p["name"] for p in description["parameters"] if p["name"].startswith("pos.")):
        generated[name] = rust_parameter(description, name, ids, tmp_path).reshape(weights[name].shape)
    alone = evaluate(description, generated, ids, tmp_path)
    gap_alone = float(np.abs(alone - expected).max())
    print(f"{label}: gap {gap:.1e} with the module's tables, {gap_alone:.1e} with the generator")
    assert gap_alone <= 1e-5, f"{label}: Rust generator {gap_alone} away from PyTorch"


def test_freezing_is_a_no_op_on_modules_without_dynamic_frequencies():
    model = Lm({"rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0, "low_freq_factor": 1.0,
                "high_freq_factor": 4.0, "original_max_position_embeddings": 4})
    # The class is decorated but recomputes nothing for Llama 3.1: freezing it changes nothing.
    before = model.rope.inv_freq.clone()
    freeze_rotary_embeddings(model, Lm.T)
    assert torch.equal(model.rope.inv_freq, before)
    assert freeze_rotary_embeddings(nn.Linear(2, 2), 4) == []
    with pytest.raises(ExportError):
        freeze_rotary_embeddings(model, 0)


TABLE_LAWS = [
    ("yarn-explicit", VARIANTS[8][1], 131072),
    ("base-10000", {"rope_type": "default", "rope_theta": 10000.0}, 8192),
    ("base-150000", {"rope_type": "default", "rope_theta": 150000.0}, 8192),
    ("base-500000", {"rope_type": "default", "rope_theta": 500000.0}, 8192),
]


@pytest.mark.parametrize("label,params,max_positions", TABLE_LAWS, ids=[t[0] for t in TABLE_LAWS])
def test_rust_tables_match_the_reference(label, params, max_positions, tmp_path):
    """The Rust generator against the tables of `LlamaRotaryEmbedding.forward`, over 2,048 positions.

    With the reference's exact frequencies (explicit form), only the gap between the cosines of two
    libraries remains: one unit in the last place. With the original law, Rust recomputes the
    frequencies with an f32 power that may differ from `torch`'s by one ulp, and the gap then grows
    with the position (1.5e-5 for base 150,000, measured on 2026-09-21).
    """
    inv_freq, attention = reference(params, max_positions, 64)
    t_len, d = 2048, HD
    if params["rope_type"] == "default":
        law = {"base": params["rope_theta"]}
    else:
        law = {"frequencies": [float(w) for w in inv_freq.tolist()], "amplitude": attention}
    desc = {
        "format": "gg-graph/v1", "name": "table",
        "inputs": [{"name": "x", "shape": [0, t_len, d]}],
        "parameters": [
            {"name": "w", "shape": [1], "init": "ones"},
            {"name": "c", "shape": [1, t_len, d], "init": {"rotary_cos": law}, "trainable": False},
            {"name": "s", "shape": [1, t_len, d], "init": {"rotary_sin": law}, "trainable": False},
        ],
        "nodes": [
            {"op": "Mul", "inputs": ["x", "w"], "output": "zero"},
            {"op": "Add", "inputs": ["zero", "c"], "output": "zc"},
            {"op": "Concat", "inputs": ["zc", "s"], "output": "out", "attrs": {"axis": 2}},
        ],
        "loss": {"mean_squared_error": {"prediction": "out",
                                        "target": {"name": "y", "shape": [0, t_len, 2 * d]}}},
    }
    model = tmp_path / "m.json"
    model.write_text(json.dumps(desc))
    inp = tmp_path / "x.ggt"
    tensors.write_tensor(inp, np.zeros((1, t_len, d), dtype=np.float32))
    out = tmp_path / "o.ggt"
    gg("job", "eval-model", str(model), "--input", f"x={inp}", "--out", str(out))
    rust = torch.from_numpy(tensors.read_tensor(out).reshape(t_len, 2 * d))
    cos, sin = reference_tables(inv_freq, attention, t_len)
    reference_both = torch.cat((cos, sin), dim=-1)
    identical = float((rust == reference_both).float().mean())
    gap = float((rust - reference_both).abs().max())
    print(f"{label}: {identical:.2%} of values byte-identical, max gap {gap:.1e}")
    # Tolerance of the driver's table recognition, max(2e-5, 5e-7·T).
    assert gap <= max(2e-5, 5e-7 * t_len), gap
    if "frequencies" in law:
        assert gap <= 2e-7, f"exact frequencies: only the cosine error should remain ({gap})"
