"""Custom operator: an element-wise expression, written inside a PyTorch module (ADR-0025).

    from gpugoblin import formula

    class Block(nn.Module):
        def forward(self, x, g):
            return formula("a * tanh(softplus(a)) + 0.1 * b", x, g)   # a = x, b = g

In PyTorch, `formula` computes the expression with torch operations (gradient included). At
capture time (`export_model`), it becomes **one** `Formula` node of the description, which the
Rust executor evaluates with the same grammar. No code is sent: only the text of the expression,
whose grammar is closed and bounded.

Grammar (identical to `crates/gg-compute/src/graph/formula.rs`):

- inputs `a`, `b`, `c`, `d` (1 to 4, all used), broadcast as for an addition;
- decimal numbers, `+ - * /`, unary minus, parentheses;
- power `x ^ p` with a **literal** exponent (`a^3`, `a^-0.5`);
- functions: exp, log, log1p, sqrt, rsqrt, abs, tanh, sigmoid, relu, sin, cos, erf, softplus,
  max(x, y), min(x, y); since `gg-ops/7`: tan, asin, acos, atan, sinh, cosh, asinh, acosh, atanh,
  atan2(y, x), floor, ceil, round (half to even), trunc, sign. The last five are piecewise
  constant: their derivative is zero wherever it exists, as in PyTorch.

Bounds: 1024 bytes, 128 terms, 32 levels of nesting.
"""

from __future__ import annotations

import math
from typing import Any, List, Sequence, Tuple

MAX_LEN = 1024
MAX_TERMS = 128
MAX_DEPTH = 32
VARIABLES = ("a", "b", "c", "d")
FUNCTIONS = {
    "exp": 1, "log": 1, "log1p": 1, "sqrt": 1, "rsqrt": 1, "abs": 1, "tanh": 1, "sigmoid": 1,
    "relu": 1, "sin": 1, "cos": 1, "erf": 1, "softplus": 1, "max": 2, "min": 2,
    # gg-ops/7 (ADR-0027)
    "tan": 1, "asin": 1, "acos": 1, "atan": 1, "sinh": 1, "cosh": 1, "asinh": 1, "acosh": 1,
    "atanh": 1, "atan2": 2, "floor": 1, "ceil": 1, "round": 1, "trunc": 1, "sign": 1,
}

#: Functions added with `gg-ops/7`: a description that uses them needs a worker at that level.
PROFILE_7_FUNCTIONS = frozenset(
    ("tan", "asin", "acos", "atan", "sinh", "cosh", "asinh", "acosh", "atanh", "atan2", "floor",
     "ceil", "round", "trunc", "sign")
)


class FormulaError(ValueError):
    """Rejected expression; the message says where, as the Rust validator does."""


# Tree: ("num", v) | ("var", i) | ("neg", e) | ("bin", op, l, r) | ("pow", e, p) | ("call", f, [args])
Expr = Tuple[Any, ...]


class _Parser:
    def __init__(self, src: str):
        self.s = src
        self.pos = 0
        self.terms = 0

    def fail(self, what: str):
        raise FormulaError(f"expression, position {self.pos}: {what}")

    def skip(self):
        while self.pos < len(self.s) and self.s[self.pos].isspace():
            self.pos += 1

    def peek(self):
        self.skip()
        return self.s[self.pos] if self.pos < len(self.s) else None

    def eat(self, c: str) -> bool:
        if self.peek() == c:
            self.pos += 1
            return True
        return False

    def term(self):
        self.terms += 1
        if self.terms > MAX_TERMS:
            self.fail(f"more than {MAX_TERMS} terms")

    def depth(self, d: int):
        if d > MAX_DEPTH:
            self.fail(f"nesting deeper than {MAX_DEPTH} levels")

    def sum(self, d: int) -> Expr:
        self.depth(d)
        left = self.product(d + 1)
        while True:
            if self.eat("+"):
                op = "+"
            elif self.eat("-"):
                op = "-"
            else:
                return left
            self.term()
            left = ("bin", op, left, self.product(d + 1))

    def product(self, d: int) -> Expr:
        self.depth(d)
        left = self.unary(d + 1)
        while True:
            if self.eat("*"):
                op = "*"
            elif self.eat("/"):
                op = "/"
            else:
                return left
            self.term()
            left = ("bin", op, left, self.unary(d + 1))

    def unary(self, d: int) -> Expr:
        self.depth(d)
        if self.eat("-"):
            self.term()
            return ("neg", self.unary(d + 1))
        return self.power(d + 1)

    def power(self, d: int) -> Expr:
        base = self.atom(d)
        if not self.eat("^"):
            return base
        self.term()
        negative = self.eat("-")
        p = self.number()
        if p is None:
            self.fail("the exponent of a power must be a literal number")
        return ("pow", base, -p if negative else p)

    def number(self):
        self.skip()
        s, start = self.s, self.pos
        p = start
        while p < len(s) and s[p].isdigit():
            p += 1
        if p < len(s) and s[p] == ".":
            p += 1
            while p < len(s) and s[p].isdigit():
                p += 1
        if p == start or (p == start + 1 and s[start] == "."):
            return None
        if p < len(s) and s[p] in "eE":
            q = p + 1
            if q < len(s) and s[q] in "+-":
                q += 1
            before = q
            while q < len(s) and s[q].isdigit():
                q += 1
            if q == before:
                self.pos = q
                self.fail("incomplete decimal exponent")
            p = q
        text = s[start:p]
        v = float(text)
        if not math.isfinite(v):
            self.fail(f"non-finite number '{text}'")
        self.pos = p
        return v

    def atom(self, d: int) -> Expr:
        self.depth(d)
        self.term()
        if self.eat("("):
            e = self.sum(d + 1)
            if not self.eat(")"):
                self.fail("expected ')'")
            return e
        v = self.number()
        if v is not None:
            return ("num", v)
        self.skip()
        start = self.pos
        while self.pos < len(self.s) and (self.s[self.pos].isascii() and (self.s[self.pos].isalnum() or self.s[self.pos] == "_")):
            self.pos += 1
        if start == self.pos:
            c = self.peek()
            self.fail(f"unexpected character '{c}'" if c is not None else "incomplete expression")
        name = self.s[start:self.pos]
        if name in VARIABLES:
            return ("var", VARIABLES.index(name))
        if name not in FUNCTIONS:
            raise FormulaError(
                f"expression: '{name}' is neither an input (a, b, c, d) nor a known function "
                f"({', '.join(FUNCTIONS)})"
            )
        if not self.eat("("):
            self.fail(f"expected '(' after {name}")
        args = [self.sum(d + 1)]
        while self.eat(","):
            args.append(self.sum(d + 1))
        if not self.eat(")"):
            self.fail("expected ')'")
        if len(args) != FUNCTIONS[name]:
            raise FormulaError(f"expression: {name} takes {FUNCTIONS[name]} argument(s), {len(args)} given")
        return ("call", name, args)


def parse(expr: str) -> Expr:
    """Parse `expr`; raise `FormulaError` with the Rust validator's message."""
    if len(expr.encode()) > MAX_LEN:
        raise FormulaError(f"expression of {len(expr.encode())} bytes, the maximum is {MAX_LEN}")
    if not expr.isascii():
        raise FormulaError("expression: only ASCII characters are accepted")
    p = _Parser(expr)
    e = p.sum(0)
    if p.peek() is not None:
        p.fail("trailing text after the expression")
    return e


def _variables(e: Expr, used: List[bool]):
    kind = e[0]
    if kind == "var":
        used[e[1]] = True
    elif kind in ("neg", "pow"):
        _variables(e[1], used)
    elif kind == "bin":
        _variables(e[2], used)
        _variables(e[3], used)
    elif kind == "call":
        for a in e[2]:
            _variables(a, used)


def check(expr: str, arity: int) -> Expr:
    """Parse `expr` for `arity` inputs and check that each one is used."""
    e = parse(expr)
    used = [False] * 4
    _variables(e, used)
    for i, u in enumerate(used):
        if u and i >= arity:
            raise FormulaError(f"expression: input '{VARIABLES[i]}' is used but the node only has {arity}")
        if not u and i < arity:
            raise FormulaError(f"expression: input '{VARIABLES[i]}' is provided but never used")
    return e


def _eval(e: Expr, ins: Sequence[Any]):
    """Evaluate the tree with torch, in the same form as the Rust executor (stable softplus,
    integer powers by repeated multiplication)."""
    import torch

    kind = e[0]
    if kind == "num":
        return e[1]
    if kind == "var":
        return ins[e[1]]
    if kind == "neg":
        return -_eval(e[1], ins)
    if kind == "bin":
        x, y = _eval(e[2], ins), _eval(e[3], ins)
        return {"+": lambda: x + y, "-": lambda: x - y, "*": lambda: x * y, "/": lambda: x / y}[e[1]]()
    if kind == "pow":
        x, p = _eval(e[1], ins), e[2]
        if not torch.is_tensor(x):
            return x ** p
        if float(p).is_integer() and abs(p) <= 8:
            n = int(abs(p))
            if n == 0:
                return torch.ones_like(x)
            acc = x
            for _ in range(n - 1):
                acc = acc * x
            return 1.0 / acc if p < 0 else acc
        return torch.pow(x, p)
    name, args = e[1], [_eval(a, ins) for a in e[2]]
    if not any(torch.is_tensor(a) for a in args):
        vals = [float(a) for a in args]
        return {
            "exp": math.exp, "log": math.log, "log1p": math.log1p, "sqrt": math.sqrt,
            "rsqrt": lambda v: 1.0 / math.sqrt(v), "abs": abs, "tanh": math.tanh,
            "sigmoid": lambda v: 1.0 / (1.0 + math.exp(-v)), "relu": lambda v: max(v, 0.0),
            "sin": math.sin, "cos": math.cos, "erf": math.erf,
            "softplus": lambda v: max(v, 0.0) + math.log1p(math.exp(-abs(v))),
            "max": max, "min": min,
            "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "sinh": math.sinh, "cosh": math.cosh, "asinh": math.asinh, "acosh": math.acosh,
            "atanh": math.atanh, "atan2": math.atan2, "floor": math.floor, "ceil": math.ceil,
            # Python's round is half to even, as torch.round.
            "round": round, "trunc": math.trunc,
            "sign": lambda v: (v > 0) - (v < 0) if v == v else v,
        }[name](*vals)
    if name == "atan2":
        y, x = args
        if not torch.is_tensor(y):
            y = torch.full_like(x, float(y))
        if not torch.is_tensor(x):
            x = torch.full_like(y, float(x))
        return torch.atan2(y, x)
    if name in ("max", "min"):
        x, y = args
        if not torch.is_tensor(x):
            x, y = y, x
        if not torch.is_tensor(y):
            return torch.clamp(x, min=y) if name == "max" else torch.clamp(x, max=y)
        return torch.maximum(x, y) if name == "max" else torch.minimum(x, y)
    x = args[0]
    if name == "softplus":
        return torch.relu(x) + torch.log1p(torch.exp(-torch.abs(x)))
    if name == "rsqrt":
        return torch.rsqrt(x)
    return getattr(torch, name)(x)


def evaluate(expr: str, *inputs):
    """Value of `expr` on torch tensors, without going through the registered operator."""
    return _eval(check(expr, len(inputs)), inputs)


_OP = None


def _op():
    """The `gpugoblin::formula` operator, registered with torch on first use: opaque to
    `torch.export` (a single node, which the driver translates into `Formula`), differentiable in
    PyTorch by recomputing the expression."""
    global _OP
    if _OP is not None:
        return _OP
    import torch

    # Explicit schema: this module's annotations are strings (`from __future__`).
    @torch.library.custom_op("gpugoblin::formula", mutates_args=(),
                             schema="(Tensor[] inputs, str expr) -> Tensor")
    def formula_op(inputs, expr):
        out = evaluate(expr, *inputs)
        return out.clone() if any(out is i for i in inputs) else out

    @formula_op.register_fake
    def _(inputs, expr):
        shape = torch.broadcast_shapes(*[i.shape for i in inputs])
        return inputs[0].new_empty(shape)

    def setup_context(ctx, inputs, output):
        ctx.saved = list(inputs[0])
        ctx.expr = inputs[1]

    def backward(ctx, grad):
        with torch.enable_grad():
            ins = [i.detach().requires_grad_(True) for i in ctx.saved]
            out = evaluate(ctx.expr, *ins)
            grads = torch.autograd.grad(out, ins, grad, allow_unused=True)
        return [g if g is not None else torch.zeros_like(i) for g, i in zip(grads, ins)], None

    formula_op.register_autograd(backward, setup_context=setup_context)
    _OP = formula_op
    return _OP


def formula(expr: str, *inputs):
    """Apply the expression `expr` to the tensors `inputs` (named `a`, `b`, `c`, `d`).

    The expression is checked immediately (grammar, bounds, every input used): an error shows up
    on the first forward pass, not on the grid.
    """
    if not 1 <= len(inputs) <= 4:
        raise FormulaError(f"formula takes 1 to 4 tensors, {len(inputs)} given")
    check(expr, len(inputs))
    return _op()(list(inputs), expr)
