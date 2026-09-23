"""Translations of `gg-ops/7` (ADR-0027): each PyTorch operator the driver now translates, captured
from a small module, validated and executed by the Rust tool, and compared with PyTorch. The
weights are perturbed first, so that a zero bias or a unit gain cannot hide an error.

The operators' own oracles (value and derivative against their f64 definition) are in
`crates/gg-compute/tests/graph_ops7.rs`; here, what is checked is that the translation computes
**the same function** as PyTorch, and that what stays outside the profile is rejected by name.
"""

import json
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gpugoblin import UnsupportedOperator, export_model, tensors  # noqa: E402
from gpugoblin.export import PROFILE_VERSION  # noqa: E402
from gg_binary import GG  # noqa: E402


def gg(*args):
    if not GG.exists():
        pytest.skip("gg binary not found (set GG_BIN)")
    out = subprocess.run([str(GG), *args], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout


def run_rust(model, x, tmp_path, dtypes=None):
    """Capture on the first two examples, check with `gg`, execute with `gg` on **all** of `x` and
    on five other examples — batch sizes other than the capture's — and return (description,
    outputs, expected), the outputs of every batch concatenated."""
    torch.manual_seed(7)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    desc, weights = export_model(model, x[:2], "mse", input_dtypes=dtypes)
    path = tmp_path / "m.json"
    path.write_text(json.dumps(desc))
    report = json.loads(gg("job", "check-model", str(path), "--json"))
    assert report["profile"] == PROFILE_VERSION
    ck = tmp_path / "ck.ggb"
    tensors.write_bundle(ck, weights)
    got, expected = [], []
    more = torch.cat([x, x[:2]]) if x.is_floating_point() is False else torch.randn((5,) + tuple(x.shape[1:]))
    for xb in (x, more):
        inp, out = tmp_path / "x.ggt", tmp_path / "o.ggt"
        tensors.write_tensor(inp, xb.numpy().astype(np.float32))
        gg("job", "eval-model", str(path), "--checkpoint", str(ck), "--input", f"x={inp}", "--out", str(out))
        got.append(tensors.read_tensor(out))
        with torch.no_grad():
            expected.append(model(xb).numpy())
    return desc, np.concatenate(got), np.concatenate(expected)


def gap(got, expected) -> float:
    assert got.shape == expected.shape, (got.shape, expected.shape)
    return float(np.abs(got - expected).max()) / max(1.0, float(np.abs(expected).max()))


class Fn(nn.Module):
    """`fn(self, x)` after a linear layer, so that the operator sees trained values."""

    def __init__(self, fn, width=16, out=None):
        super().__init__()
        self.lin = nn.Linear(width, out or width)
        self.fn = fn
        self.register_buffer("idx", torch.tensor([3, 0, 3, 5, 1]))
        self.register_buffer("dst", torch.tensor([1, 1, 2, 7, 0]))
        self.register_buffer("g2", torch.tensor([[0, 2, 3], [0, 2, 3]]))

    def forward(self, x):
        return self.fn(self, self.lin(x))


def seq(x, c=2):
    """[N, 16] -> [N, c, 16 / c]: a channel axis for the 1D operators."""
    return x.reshape(x.shape[0], c, -1)


def img(x):
    """[N, 16] -> [N, 1, 4, 4]."""
    return x.reshape(x.shape[0], 1, 4, 4)


ELEMENTWISE = [
    ("tan", lambda m, x: torch.tan(x * 0.5)),
    ("asin acos", lambda m, x: torch.asin(torch.tanh(x) * 0.9) + torch.acos(torch.sigmoid(x))),
    ("atan sinh cosh", lambda m, x: torch.atan(x) + torch.sinh(x * 0.5) - torch.cosh(x * 0.3)),
    ("asinh acosh atanh", lambda m, x: torch.asinh(x) + torch.acosh(x * x + 1.5) + torch.atanh(torch.tanh(x) * 0.7)),
    ("atan2", lambda m, x: torch.atan2(x, x.flip(-1) + 0.3) + torch.atan2(x, torch.tensor(2.0))),
    ("floor ceil trunc round sign", lambda m, x: torch.floor(x * 3) + torch.ceil(x * 2) + torch.trunc(x * 5)
     + torch.round(x * 4) + torch.sign(x) + x),
    ("round decimals", lambda m, x: torch.round(x, decimals=2) + torch.round(x * 100, decimals=-1)),
    ("log2 log10 exp2", lambda m, x: torch.log2(x.abs() + 1) + torch.log10(x.abs() + 2) + torch.exp2(x)),
    ("remainder fmod", lambda m, x: torch.remainder(x * 3, 0.7) + torch.fmod(x * 3, -0.7)
     + torch.remainder(x, x.flip(-1).abs() + 0.5)),
    ("floor_divide and rounding division", lambda m, x: torch.floor_divide(x * 5, 2.0)
     + torch.div(x * 5, 3.0, rounding_mode="trunc") + torch.div(x * 7, x.abs() + 1, rounding_mode="floor")),
    ("frac erfc softsign tanhshrink selu celu", lambda m, x: torch.frac(x * 3) + torch.special.erfc(x)
     + F.softsign(x) + F.tanhshrink(x) + F.selu(x) + F.celu(x, 0.5)),
    ("clamp by tensors, lerp, addcmul, addcdiv, add alpha", lambda m, x: torch.clamp(x, -x.flip(-1).abs(), x.flip(-1).abs() + 0.1)
     + torch.lerp(x, x.flip(-1), 0.25) + torch.lerp(x, x.flip(-1), 0.75) + torch.addcmul(x, x, x, value=0.5)
     + torch.addcdiv(x, x, x.abs() + 1, value=-2.0) + torch.add(x, x.flip(-1), alpha=3.0)),
    ("hypot logaddexp logsigmoid", lambda m, x: torch.hypot(x, x.flip(-1)) + torch.logaddexp(x, x.flip(-1))
     + F.logsigmoid(x)),
]

REDUCTIONS = [
    ("sum", lambda m, x: seq(x).sum(-1) + seq(x).sum(dim=(1, 2), keepdim=True).squeeze(-1)),
    ("amax amin", lambda m, x: seq(x).amax(-1) - seq(x).amin(dim=1).sum(-1, keepdim=True)),
    ("max.dim min.dim values", lambda m, x: seq(x).max(dim=-1).values + seq(x).min(dim=2, keepdim=True)[0].squeeze(-1)),
    ("var std", lambda m, x: seq(x).var(-1) + seq(x).std(dim=-1, unbiased=False) + seq(x).var(dim=(1, 2), keepdim=True).squeeze(-1)),
    ("var_mean", lambda m, x: torch.cat(torch.var_mean(seq(x), dim=-1), -1)),
    ("prod", lambda m, x: seq(x, 4).prod(-1)),
    ("cumsum", lambda m, x: torch.cumsum(x, dim=1)),
    ("logsumexp log_softmax", lambda m, x: torch.logsumexp(seq(x), dim=-1) + F.log_softmax(seq(x), dim=-1).sum(-1)),
    ("vector norms", lambda m, x: torch.linalg.vector_norm(seq(x), dim=-1) + seq(x).norm(p=1, dim=-1)
     + F.normalize(seq(x), dim=-1).sum(-1)),
]

SHAPES = [
    ("constant pad", lambda m, x: F.pad(seq(x), (2, 1), value=0.5).sum(-1)),
    ("constant pad 2D with a crop", lambda m, x: F.pad(img(x), (1, -1, 0, 2)).flatten(1)),
    ("reflection pad 1D", lambda m, x: F.pad(seq(x), (3, 2), mode="reflect").flatten(1)),
    ("reflection pad 2D", lambda m, x: F.pad(img(x), (1, 2, 2, 1), mode="reflect").flatten(1)),
    ("replication pad", lambda m, x: F.pad(seq(x), (1, 3), mode="replicate").flatten(1)
     + F.pad(img(x), (2, 0, 1, 1), mode="replicate").flatten(1)[:, :10].sum(-1, keepdim=True)),
    ("circular pad", lambda m, x: F.pad(seq(x), (2, 3), mode="circular").flatten(1)),
    ("upsample nearest", lambda m, x: F.interpolate(img(x), scale_factor=2, mode="nearest").flatten(1)
     + F.interpolate(seq(x), size=10, mode="nearest").flatten(1).sum(-1, keepdim=True)),
    ("upsample nearest to a size", lambda m, x: F.interpolate(img(x), size=(7, 5), mode="nearest").flatten(1)),
    ("upsample bilinear", lambda m, x: F.interpolate(img(x), scale_factor=2, mode="bilinear").flatten(1)),
    ("upsample bilinear aligned", lambda m, x: F.interpolate(img(x), size=(6, 9), mode="bilinear", align_corners=True).flatten(1)),
    ("upsample linear 1D", lambda m, x: F.interpolate(seq(x), scale_factor=3, mode="linear").flatten(1)),
    ("downsample bilinear", lambda m, x: F.interpolate(img(x), size=(3, 2), mode="bilinear").flatten(1)),
    ("flip roll repeat", lambda m, x: seq(x).flip(-1).flatten(1) + torch.roll(seq(x), shifts=(1, -3), dims=(1, 2)).flatten(1)
     + seq(x).repeat(1, 1, 2)[:, :, 3:13].flatten(1).sum(-1, keepdim=True)),
]

INDEXING = [
    ("index_select by a buffer", lambda m, x: torch.index_select(x, 1, m.idx)),
    ("indexing by a buffer", lambda m, x: x[:, m.idx] + seq(x)[:, :, m.idx[:3]].flatten(1).sum(-1, keepdim=True)),
    ("index_add", lambda m, x: torch.zeros_like(x).index_add(1, m.dst, x[:, m.idx]) + x),
    ("index_add alpha", lambda m, x: x.index_add(1, m.dst, x[:, m.idx], alpha=-0.5)),
    ("gather along the broadcast of a vector", lambda m, x: torch.gather(seq(x), 2, m.g2.unsqueeze(0).expand(x.shape[0], 2, 3)).flatten(1)),
    ("scatter_add", lambda m, x: torch.zeros_like(seq(x)).scatter_add(2, m.g2.unsqueeze(0).expand(x.shape[0], 2, 3), seq(x)[:, :, :3]).flatten(1)),
]


@pytest.mark.parametrize("label,fn", ELEMENTWISE + REDUCTIONS + SHAPES + INDEXING,
                         ids=[c[0] for c in ELEMENTWISE + REDUCTIONS + SHAPES + INDEXING])
def test_the_translation_computes_the_same_function(label, fn, tmp_path):
    model = Fn(fn).eval()
    x = torch.randn(3, 16)
    _, got, expected = run_rust(model, x, tmp_path)
    assert gap(got, expected) <= 1e-5, f"{label}: gap {gap(got, expected)}"


class Vocoder(nn.Module):
    """HiFi-GAN in miniature: transposed 1D convolutions, strided and grouped, then a 2D one."""

    def __init__(self):
        super().__init__()
        self.pre = nn.Conv1d(4, 8, 3, padding=1)
        self.up1 = nn.ConvTranspose1d(8, 6, 8, stride=4, padding=2)
        self.up2 = nn.ConvTranspose1d(6, 4, 3, stride=2, padding=1, output_padding=1, groups=2, dilation=2)
        self.up2d = nn.ConvTranspose2d(1, 2, (3, 2), stride=(2, 1), padding=(1, 0), output_padding=(1, 0))

    def forward(self, x):
        h = self.up2(F.leaky_relu(self.up1(F.leaky_relu(self.pre(x), 0.1)), 0.1))
        return torch.tanh(self.up2d(h[:, :1].unsqueeze(1)).flatten(1))


def test_transposed_convolutions(tmp_path):
    desc, got, expected = run_rust(Vocoder().eval(), torch.randn(2, 4, 5), tmp_path)
    assert sum(n["op"] == "ConvTranspose2d" for n in desc["nodes"]) == 3
    assert gap(got, expected) <= 1e-5


class ShortKernels(nn.Module):
    """Kernels shorter than their stride: a ResNet's 1×1 stride-2 shortcut, and transposed
    convolutions that leave gaps. The executor rewrites them (Burn's GPU kernel is wrong there);
    the rewrite must compute PyTorch's function, paddings and groups included."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 8, 1, stride=2)
        self.c2 = nn.Conv2d(8, 8, (2, 1), stride=(3, 2), padding=(1, 0), groups=2)
        self.t1 = nn.ConvTranspose2d(8, 6, 1, stride=2, output_padding=1)
        self.t2 = nn.ConvTranspose2d(6, 4, (2, 3), stride=(3, 2), padding=(1, 1), output_padding=(2, 1), groups=2)
        self.s1 = nn.Conv1d(4, 4, 1, stride=3, padding=2)
        self.u1 = nn.ConvTranspose1d(4, 2, 2, stride=5, padding=1, output_padding=3)

    def forward(self, x):
        h = self.t2(F.relu(self.t1(F.relu(self.c2(F.relu(self.c1(x)))))))
        v = self.u1(self.s1(h.mean(2)))
        return torch.cat([h.flatten(1), v.flatten(1)], 1)


def test_kernels_shorter_than_their_stride(tmp_path):
    desc, got, expected = run_rust(ShortKernels().eval(), torch.randn(2, 3, 13, 11), tmp_path)
    assert sum(n["op"] == "ConvTranspose2d" for n in desc["nodes"]) == 3
    assert gap(got, expected) <= 1e-5


class WeightNormed(nn.Module):
    """Weight normalization, as Wav2Vec2 and the vocoders use it: the weight is recomputed from
    `v` and `g` at every forward pass (`_weight_norm`)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.utils.parametrizations.weight_norm(nn.Conv1d(4, 6, 3, padding=1), dim=2)
        self.lin = nn.utils.parametrizations.weight_norm(nn.Linear(6, 5))

    def forward(self, x):
        return self.lin(self.conv(x).mean(-1))


def test_weight_normalization(tmp_path):
    desc, got, expected = run_rust(WeightNormed().eval(), torch.randn(3, 4, 7), tmp_path)
    names = {p["name"] for p in desc["parameters"]}
    # `g` and `v` are the trained parameters, as in PyTorch.
    assert any("original0" in n for n in names) and any("original1" in n for n in names)
    assert gap(got, expected) <= 1e-5


class RelativeBias(nn.Module):
    """A relative position bias in the style of T5: buckets computed from constants, then an embedding
    looked up with that constant, non-contiguous index."""

    def __init__(self, t=6, buckets=5, heads=2):
        super().__init__()
        self.emb = nn.Embedding(buckets, heads)
        pos = torch.arange(t)
        rel = (pos[None, :] - pos[:, None]).clamp(-2, 2) + 2
        self.register_buffer("bucket", rel)
        self.lin = nn.Linear(t, t)

    def forward(self, x):  # x: [N, heads, t, t]
        bias = self.emb(self.bucket).permute(2, 0, 1).unsqueeze(0)
        return self.lin(x + bias).softmax(-1)


def test_an_embedding_with_a_constant_non_contiguous_index(tmp_path):
    desc, got, expected = run_rust(RelativeBias().eval(), torch.randn(2, 2, 6, 6), tmp_path)
    assert any(n["op"] == "IndexSelect" for n in desc["nodes"])
    assert gap(got, expected) <= 1e-5


# argmax, topk, sort and indices computed from the data are in the profile since gg-ops/8:
# tests/test_ops8.py.
REFUSED = [
    ("gather with a varying index", lambda m, x: torch.gather(seq(x), 2, torch.tensor([[[0, 1], [1, 0]]]).expand(x.shape[0], 2, 2)), "varies"),
    ("sum over the batch", lambda m, x: x.sum(0, keepdim=True) + x, "batch"),
    ("index_select along the batch", lambda m, x: torch.index_select(x, 0, torch.tensor([0, 0, 1])), "batch"),
    ("upsample with a non-integer scale", lambda m, x: F.interpolate(seq(x), scale_factor=1.7, mode="nearest").flatten(1), "scale factor"),
    ("nearest-exact", lambda m, x: F.interpolate(seq(x), scale_factor=2, mode="nearest-exact").flatten(1), "nearest"),
]


@pytest.mark.parametrize("label,fn,reason", REFUSED, ids=[c[0] for c in REFUSED])
def test_what_stays_outside_is_rejected_by_name(label, fn, reason):
    with pytest.raises(UnsupportedOperator) as e:
        export_model(Fn(fn).eval(), torch.randn(3, 16), "mse")
    assert reason in str(e.value), str(e.value)
