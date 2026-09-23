"""`gpugoblin.formula`: the custom operator written inside a PyTorch module (ADR-0025).

Three proofs: the Python grammar rejects what the Rust validator rejects, with the same message;
in PyTorch, value and gradient follow the definition; once captured, the expression becomes a
single `Formula` node that the Rust executor computes as PyTorch does.
"""

import json
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from gpugoblin import FormulaError, export_model, formula, tensors  # noqa: E402
from gpugoblin.formula import check, evaluate  # noqa: E402
from gg_binary import GG  # noqa: E402


@pytest.mark.parametrize("expr,arity,message", [
    ("a +", 1, "incomplete"),
    ("pow(a, 2)", 1, "neither an input"),
    ("max(a)", 1, "2 argument"),
    ("a ^ b", 2, "literal number"),
    ("(a", 1, "')'"),
    ("a b", 1, "trailing text"),
    ("é", 1, "ASCII"),
    ("a + b + c", 2, "only has 2"),
    ("a * 2", 2, "never used"),
    (" + ".join(["a"] * 200), 1, "terms"),
    ("(" * 40 + "a" + ")" * 40, 1, "nesting"),
    ("1e999 * a", 1, "non-finite"),
    ("atan2(a)", 1, "2 argument"),
    ("floor(a, a)", 1, "1 argument"),
])
def test_the_grammar_rejects_like_the_rust_validator(expr, arity, message):
    with pytest.raises(FormulaError) as e:
        check(expr, arity)
    assert message in str(e.value)


def test_value_and_gradient_follow_the_definition():
    torch.manual_seed(0)
    x = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    g = torch.randn(7, dtype=torch.float64, requires_grad=True)
    y = formula("a * tanh(softplus(a)) + b^3 - max(a, 0.5) / (1 + b * b)", x, g)
    ref = torch.nn.functional.mish(x) + g ** 3 - torch.clamp(x, min=0.5) / (1 + g * g)
    assert torch.allclose(y, ref, atol=1e-12)
    gy = torch.randn_like(y)
    got = torch.autograd.grad(y, (x, g), gy)
    want = torch.autograd.grad(ref, (x, g), gy)
    for a, b in zip(got, want):
        assert torch.allclose(a, b, atol=1e-10)
    # Integer power of a negative base: exact, not NaN.
    assert float(evaluate("a^3", torch.tensor([-2.0]))) == -8.0


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(6, 8)
        self.g = nn.Parameter(torch.randn(8))
        self.out = nn.Linear(8, 3)

    def forward(self, x):
        h = formula("a * sigmoid(b) + erf(a) * 0.1 - log1p(abs(b))", self.lin(x), self.g)
        return self.out(h)


def gg(*args):
    if not GG.exists():
        pytest.skip("gg binary not found (set GG_BIN)")
    out = subprocess.run([str(GG), *args], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_a_custom_operator_is_one_node_computed_like_pytorch(tmp_path):
    torch.manual_seed(1)
    model = Block().eval()
    x = torch.randn(4, 6)
    desc, weights = export_model(model, x, "mse")
    nodes = [n for n in desc["nodes"] if n["op"] == "Formula"]
    assert len(nodes) == 1
    assert nodes[0]["attrs"]["expr"] == "a * sigmoid(b) + erf(a) * 0.1 - log1p(abs(b))"
    path = tmp_path / "m.json"
    path.write_text(json.dumps(desc))
    report = json.loads(gg("job", "check-model", str(path), "--json"))
    from gpugoblin.export import PROFILE_VERSION

    assert report["profile"] == PROFILE_VERSION
    ck, inp, out = tmp_path / "ck.ggb", tmp_path / "x.ggt", tmp_path / "o.ggt"
    tensors.write_bundle(ck, weights)
    tensors.write_tensor(inp, x.numpy())
    gg("job", "eval-model", str(path), "--checkpoint", str(ck), "--input", f"x={inp}", "--out", str(out))
    with torch.no_grad():
        expected = model(x).numpy()
    gap = float(np.abs(tensors.read_tensor(out) - expected).max())
    assert gap <= 1e-6, gap


EXPRESSIONS = [
    "exp(a) - log(abs(a) + 1)", "sqrt(abs(a) + 0.5) * rsqrt(abs(a) + 2)", "tanh(a) * sigmoid(a)",
    "relu(a) + sin(a) * cos(a)", "erf(a) + softplus(a * 30) / 30", "max(a, -0.2) + min(a, 0.3)",
    # Negative power kept away from zero: `a` comes out of a matrix product whose rounding differs
    # by 1e-7 between torch and Rust, and `1/a²` near zero would amplify that input gap (measured:
    # 9e-5 at a = 0.002, with or without `^`), which would say nothing about the operator.
    "a^3 - (a * a + 1)^-2 + (abs(a) + 1)^0.75", "-(a - 1) / (a * a + 1)",
    # gg-ops/7 (ADR-0027)
    "tan(a * 0.3) + atan(a) - asin(tanh(a) * 0.9) + acos(sigmoid(a))",
    "sinh(a * 0.5) - cosh(a * 0.3) + asinh(a) + acosh(a * a + 1.5) + atanh(tanh(a) * 0.7)",
    "atan2(a, abs(a) + 0.5) + atan2(1, a) + atan2(a, -2)",
    "floor(a * 3) + ceil(a * 2) + round(a * 4) + trunc(a * 5) + sign(a) + a",
]


@pytest.mark.parametrize("expr", EXPRESSIONS)
def test_python_and_rust_agree_on_every_function(expr, tmp_path):
    """The same expression, evaluated by torch and by the Rust executor, on the same values."""
    torch.manual_seed(2)

    class One(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(5, 5)

        def forward(self, x):
            return formula(expr, self.lin(x) * 3)

    model = One().eval()
    x = torch.randn(8, 5)
    desc, weights = export_model(model, x, "mse")
    path = tmp_path / "m.json"
    path.write_text(json.dumps(desc))
    ck, inp, out = tmp_path / "ck.ggb", tmp_path / "x.ggt", tmp_path / "o.ggt"
    tensors.write_bundle(ck, weights)
    tensors.write_tensor(inp, x.numpy())
    gg("job", "eval-model", str(path), "--checkpoint", str(ck), "--input", f"x={inp}", "--out", str(out))
    with torch.no_grad():
        expected = model(x).numpy()
    got = tensors.read_tensor(out)
    gap = float(np.abs(got - expected).max()) / max(1.0, float(np.abs(expected).max()))
    assert gap <= 2e-6, f"{expr}: gap {gap}"
