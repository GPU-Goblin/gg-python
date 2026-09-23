"""Translations of `gg-ops/8` (ADR-0029): integer tensors computed inside the graph. Each construct
is captured from a small module, validated and executed by the Rust tool, and compared with
PyTorch — the values that flow through the indices, and the indices themselves once converted to
f32. What stays outside the profile is rejected by name.

The operators' own oracles (exact indices, value and derivative against an f64 definition) are in
`crates/gg-compute/tests/graph_ops8.rs`.
"""

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gpugoblin import UnsupportedOperator, export_model  # noqa: E402
from test_ops7 import Fn, gap, run_rust, seq  # noqa: E402


def as_f(i):
    return i.float()


CASES = [
    ("argmax argmin", lambda m, x: torch.cat([as_f(x.argmax(-1, keepdim=True)), as_f(seq(x).argmin(dim=2)), x], -1)),
    ("max(dim) values and indices", lambda m, x: torch.cat([seq(x).max(dim=-1).values, as_f(seq(x).max(dim=-1).indices),
                                                            seq(x).min(dim=1, keepdim=True).values.flatten(1)], -1)),
    ("topk values and indices", lambda m, x: torch.cat([x.topk(3).values, as_f(x.topk(3).indices),
                                                         seq(x, 4).topk(2, dim=1, largest=False).values.flatten(1)], -1)),
    ("sort and argsort", lambda m, x: torch.cat([x.sort(dim=-1, descending=True).values, as_f(x.argsort(-1)),
                                                  torch.sort(seq(x), dim=-1, stable=True).values.flatten(1)], -1)),
    ("gather at a computed index", lambda m, x: torch.gather(x, 1, x.topk(4).indices) * 2 + torch.gather(seq(x), 2, seq(x).argsort(-1)).flatten(1)[:, :4]),
    ("scatter of distinct positions", lambda m, x: torch.zeros_like(x).scatter(-1, x.topk(3).indices, x.topk(3).values.softmax(-1)) + x),
    ("scatter of a value", lambda m, x: x.scatter(1, x.topk(2).indices, 0.5)),
    ("scatter_add at a computed index", lambda m, x: x.scatter_add(1, x.argmax(-1, keepdim=True).expand(-1, 3), x[:, :3])),
    ("scatter_reduce sum and mean", lambda m, x: x.scatter_reduce(1, x.argmin(-1, keepdim=True).expand(-1, 4), x[:, 4:8], "sum")
     + x.scatter_reduce(1, x.argmax(-1, keepdim=True).expand(-1, 3), x[:, :3], "mean", include_self=False)
     + x.scatter_reduce(1, x.argmax(-1, keepdim=True).expand(-1, 2), x[:, 5:7], "mean")),
    ("repeat_interleave and integer reshape", lambda m, x: seq(x).repeat_interleave(2, dim=1).flatten(1)
     + as_f(x.topk(2).indices.reshape(x.shape[0], 2).repeat_interleave(16, dim=1)).flatten(1)),
]


@pytest.mark.parametrize("label,fn", CASES, ids=[c[0] for c in CASES])
def test_the_translation_computes_the_same_function(label, fn, tmp_path):
    model = Fn(fn).eval()
    x = torch.randn(3, 16)
    _, got, expected = run_rust(model, x, tmp_path)
    assert gap(got, expected) <= 1e-5, f"{label}: gap {gap(got, expected)}"


class Router(nn.Module):
    """Top-2 routing to experts whose weights are gathered per token (transformers' batched_mm
    experts, in miniature): an integer reshape, a clamp that changes nothing, a gather of rows of a
    rank-3 parameter, batched products, and the weighted sum."""

    def __init__(self, experts=4, dim=8, hidden=6):
        super().__init__()
        self.router = nn.Linear(dim, experts, bias=False)
        self.up = nn.Parameter(torch.randn(experts, hidden, dim) * 0.3)
        self.down = nn.Parameter(torch.randn(experts, dim, hidden) * 0.3)
        self.experts = experts

    def forward(self, x):  # [batch, T, dim]
        b, t, d = x.shape
        h = x.reshape(-1, d)
        w, idx = torch.topk(self.router(h).softmax(-1), 2, dim=-1)
        w = w / w.sum(-1, keepdim=True)
        rep = h.repeat_interleave(2, dim=0)
        ids = idx.reshape(-1).clamp(0, self.experts - 1)
        up = torch.bmm(self.up[ids], rep.unsqueeze(-1)).squeeze(-1)
        down = torch.bmm(self.down[ids], F.silu(up).unsqueeze(-1)).squeeze(-1)
        out = (down * w.reshape(-1, 1)).view(b * t, 2, d).sum(1)
        return out.view(b, t, d)


def test_a_router_with_gathered_expert_weights(tmp_path):
    desc, got, expected = run_rust(Router().eval(), torch.randn(2, 5, 8), tmp_path)
    ops = {n["op"] for n in desc["nodes"]}
    assert {"ArgTopK", "GatherElements", "Gather"} <= ops, ops
    assert gap(got, expected) <= 1e-5


class ArgmaxPooling(nn.Module):
    """An index read by an embedding: the table's rows chosen by the argmax of a projection."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 5)
        self.emb = nn.Embedding(5, 3)

    def forward(self, x):
        return self.emb(self.lin(x).argmax(-1))


def test_an_embedding_read_at_a_computed_index(tmp_path):
    desc, got, expected = run_rust(ArgmaxPooling().eval(), torch.randn(4, 8), tmp_path)
    assert any(n["op"] == "ArgMax" for n in desc["nodes"])
    assert gap(got, expected) <= 1e-6


REFUSED = [
    ("argmax over everything", lambda m, x: as_f(x.argmax()).expand(x.shape[0], 1), "batch"),
    ("topk unsorted", lambda m, x: x.topk(2, sorted=False).values, "sorted=False"),
    ("integer arithmetic", lambda m, x: as_f(x.argmax(-1, keepdim=True) * 3 + 1), "integer arithmetic"),
    ("scatter_reduce amax", lambda m, x: x.scatter_reduce(1, x.argmax(-1, keepdim=True).expand(-1, 2), x[:, :2], "amax"), "amax"),
    ("one_hot", lambda m, x: F.one_hot(x.argmax(-1), 16).float(), "one_hot"),
    ("scatter of repeated positions", lambda m, x: x.scatter(1, x.argmax(-1, keepdim=True).expand(-1, 2), x[:, :2]), "distinct"),
    ("indexing along the batch by a computed index", lambda m, x: x[x[:, :2].argmax(-1)], "batch"),
]


@pytest.mark.parametrize("label,fn,reason", REFUSED, ids=[c[0] for c in REFUSED])
def test_what_stays_outside_is_rejected_by_name(label, fn, reason, tmp_path):
    from gpugoblin import ExportError

    with pytest.raises((UnsupportedOperator, ExportError, RuntimeError)) as e:
        # A rejection at capture or at validation by the Rust tool: both name the reason.
        run_rust(Fn(fn).eval(), torch.randn(3, 16), tmp_path)
    assert reason in str(e.value), str(e.value)


def test_mixtral_needs_batched_experts_and_then_captures():
    transformers = pytest.importorskip("transformers")
    from gpugoblin import batched_experts, coverage

    cfg = transformers.MixtralConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                     num_key_value_heads=1, intermediate_size=16, num_local_experts=4,
                                     num_experts_per_tok=2)
    model = transformers.MixtralForCausalLM(cfg).eval()

    class Logits(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(input_ids=x, use_cache=False).logits

    ids = torch.randint(0, 32, (2, 6))
    assert not coverage(Logits(model), ids, input_dtypes=["i64"])["captured"]
    assert batched_experts(model)
    assert coverage(Logits(model), ids, input_dtypes=["i64"])["captured"]
