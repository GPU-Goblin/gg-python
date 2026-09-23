"""Capture of a PyTorch module as a `gg-graph/v1` description.

The whole graph is captured with `torch.export`, then translated operator by operator into the
published profile (`PROFILE_VERSION`). An operator outside the profile makes the capture fail, by
name: the platform does not "support PyTorch", it supports a documented profile, and says so
before anything is sent.

What comes out of here: a description (JSON, plain data) and the module's actual weights (the
initial checkpoint, a GGB1). No code is serialized.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FORMAT = "gg-graph/v1"
PROFILE_VERSION = "gg-ops/8"

#: Below this value, a mask fill counts as "−∞" for the Softmax that follows.
MASK_FILL_THRESHOLD = -1.0e4

#: Profile operator -> ONNX equivalent, as published by `GET /v1/catalog/ops`.
PROFILE_ONNX_EQUIVALENTS = {
    "MatMul": "MatMul (opset 13)", "Add": "Add (opset 14)", "Sub": "Sub (opset 14)",
    "Mul": "Mul (opset 14)", "Div": "Div (opset 14)", "Relu": "Relu (opset 14)",
    "Gelu": "Gelu (opset 20)", "Sigmoid": "Sigmoid (opset 13)", "Tanh": "Tanh (opset 13)",
    "Softmax": "Softmax (opset 13)", "LayerNorm": "LayerNormalization (opset 17)",
    "RMSNorm": "RMSNormalization (opset 23)", "Conv2d": "Conv (opset 11, 2D)",
    "MaxPool2d": "MaxPool (opset 12, 2D)", "AvgPool2d": "AveragePool (opset 11, 2D)",
    "BatchNorm2d": "BatchNormalization (opset 15)", "Reshape": "Reshape (opset 14)",
    "Transpose": "Transpose (opset 13)", "Flatten": "Flatten (opset 13)",
    "Concat": "Concat (opset 13)", "Gather": "Gather (opset 13)", "Dropout": "Dropout (opset 13)",
    "CausalMask": "Trilu (opset 14) + Where (opset 16)",
    "Slice": "Slice (opset 13)",
    "ReduceMean": "ReduceMean (opset 18)",
    "Rsqrt": "Sqrt (opset 13) + Reciprocal (opset 13)",
    "Formula": None,
    # gg-ops/7 (ADR-0027)
    "ConvTranspose2d": "ConvTranspose (opset 11, 2D)", "Pad": "Pad (opset 18)",
    "Resize": "Resize (opset 19)", "Flip": "Slice with step -1 (opset 13)",
    "ReduceSum": "ReduceSum (opset 13)", "ReduceMax": "ReduceMax (opset 18)",
    "ReduceMin": "ReduceMin (opset 18)", "ReduceProd": "ReduceProd (opset 18)",
    "CumSum": "CumSum (opset 14)", "IndexSelect": "Gather (opset 13), constant indices",
    "IndexAdd": "ScatterElements (opset 18, reduction add), constant indices",
    # gg-ops/8 (ADR-0029)
    "ArgMax": "ArgMax (opset 13, select_last_index = 0)",
    "ArgMin": "ArgMin (opset 13, select_last_index = 0)",
    "ArgTopK": "TopK (opset 11), indices output", "ArgSort": None, "Cast": "Cast (opset 13)",
    "GatherElements": "GatherElements (opset 13)", "ScatterElements": "ScatterElements (opset 18)",
}

#: Element-wise aten functions translated into `Formula` (ADR-0025): aten name -> expression in `a`.
ELEMENTWISE_FORMULAS = {
    "exp": "exp(a)", "log": "log(a)", "log1p": "log1p(a)", "sqrt": "sqrt(a)", "abs": "abs(a)",
    "sin": "sin(a)", "cos": "cos(a)", "erf": "erf(a)", "reciprocal": "1 / a", "square": "a * a",
    "expm1": "exp(a) - 1", "zeros_like": "a * 0", "ones_like": "a * 0 + 1",
    "mish": "a * tanh(softplus(a))", "hardsigmoid": "min(max(a / 6 + 0.5, 0), 1)",
    "hardswish": "a * min(max(a / 6 + 0.5, 0), 1)", "relu6": "min(relu(a), 6)",
    # gg-ops/7 (ADR-0027). The rounding functions and `sign` are piecewise constant: their
    # derivative is zero, in PyTorch as in the executor.
    "tan": "tan(a)", "asin": "asin(a)", "acos": "acos(a)", "atan": "atan(a)", "sinh": "sinh(a)",
    "cosh": "cosh(a)", "asinh": "asinh(a)", "acosh": "acosh(a)", "atanh": "atanh(a)",
    "floor": "floor(a)", "ceil": "ceil(a)", "round": "round(a)", "trunc": "trunc(a)",
    "fix": "trunc(a)", "sign": "sign(a)", "sgn": "sign(a)", "frac": "a - trunc(a)",
    # log2 and log10 through the natural logarithm: within two ulps of PyTorch's.
    "log2": "log(a) * 1.4426950408889634", "log10": "log(a) * 0.4342944819032518",
    "exp2": "exp(a * 0.6931471805599453)", "erfc": "1 - erf(a)", "special_erfc": "1 - erf(a)",
    "softsign": "a / (1 + abs(a))", "tanhshrink": "a - tanh(a)",
    "deg2rad": "a * 0.017453292519943295", "rad2deg": "a * 57.29577951308232",
    "selu": "1.0507009873554805 * (relu(a) + min(1.6732632423543772 * (exp(a) - 1), 0))",
}

#: Binary element-wise aten functions translated into `Formula`: `{a}` and `{b}` are replaced by
#: the inputs, or by the literal when PyTorch passes a Python number. `remainder` and `fmod` are
#: written `a − b·floor(a/b)` and `a − b·trunc(a/b)`: exact when the quotient is, otherwise within
#: one unit in the last place of `a` (ADR-0027); their derivative is PyTorch's.
BINARY_FORMULAS = {
    "atan2": "atan2({a}, {b})", "arctan2": "atan2({a}, {b})",
    "remainder": "{a} - {b} * floor({a} / {b})", "fmod": "{a} - {b} * trunc({a} / {b})",
    "floor_divide": "floor({a} / {b})", "hypot": "sqrt({a} * {a} + {b} * {b})",
    "logaddexp": "max({a}, {b}) + log1p(exp(-abs({a} - {b})))",
}

#: Position tables the platform can generate (ADR-0021), recognized by their value.
POSITION_TABLES = ("rotary_cos", "rotary_sin", "sinusoidal")


def position_table(kind: str, base: float, t_len: int, d: int):
    """Table `[T, d]` generated exactly as the Rust executor does, in float32."""
    import torch

    i = torch.arange(d // 2, dtype=torch.float32)
    freqs = 1.0 / (torch.tensor(base, dtype=torch.float32) ** (2.0 * i / d))
    t = torch.arange(t_len, dtype=torch.float32)
    if kind in ("rotary_cos", "rotary_sin"):
        angle = torch.outer(t, torch.cat([freqs, freqs]))
        return angle.cos() if kind == "rotary_cos" else angle.sin()
    angle = torch.outer(t, freqs)
    out = torch.empty(t_len, d, dtype=torch.float32)
    out[:, 0::2] = angle.sin()
    out[:, 1::2] = angle.cos()
    return out


def _snap(base: float) -> list:
    """The estimate, and its value rounded to two significant digits: the usual bases are round
    (10,000, 500,000, 1,000,000) and estimating from a cosine is ill-conditioned."""
    import math

    if not (base > 1.0 and math.isfinite(base)):
        return []
    k = math.floor(math.log10(base))
    step = 10.0 ** (k - 1)
    rounded = round(base / step) * step
    return [rounded, base] if abs(rounded - base) > 0 else [base]


def recognise_position_table(value):
    """`{kind: {"base": b}}` if `value` is a position table the platform generates, up to f32
    rounding; otherwise None. Expected shape: `[1, …, 1, T, d]`, d even."""
    import math

    import torch

    v = value.detach().to(torch.float32)
    if v.dim() < 2 or any(s != 1 for s in v.shape[:-2]):
        return None
    t_len, d = int(v.shape[-2]), int(v.shape[-1])
    if d < 4 or d % 2 or t_len < 2:
        return None
    tab = v.reshape(t_len, d)
    half = d // 2
    # Tolerance: the angle t·ω is computed in f32, its error grows with the position.
    tol = max(2e-5, 5e-7 * t_len)
    for kind in POSITION_TABLES:
        # At position 1, the angle is the frequency itself: the base is derived from it.
        if kind == "rotary_sin":
            x = float(tab[1, half - 1])
            freq, exponent = (math.asin(x), d / (d - 2)) if abs(x) <= 1 else (None, None)
        elif kind == "sinusoidal":
            x = float(tab[1, d - 2])
            freq, exponent = (math.asin(x), d / (d - 2)) if abs(x) <= 1 else (None, None)
        else:
            x = float(tab[1, 1])
            freq, exponent = (math.acos(x), d / 2) if abs(x) <= 1 else (None, None)
        if not freq or freq <= 0:
            continue
        candidates = _snap(freq ** (-exponent)) + [10000.0, 500000.0, 1000000.0]
        for base in candidates:
            if (position_table(kind, base, t_len, d) - tab).abs().max() <= tol:
                return {kind: {"base": float(base)}}
    return None


def rotary_table(kind: str, frequencies, amplitude: float, t_len: int):
    """Rotary table `[T, d]` with explicit frequencies, computed as the Rust executor does:
    `A·cos(t·ω)` in float32, "rotate_half" layout (ADR-0022)."""
    import torch

    freqs = torch.as_tensor(frequencies, dtype=torch.float64).to(torch.float32)
    t = torch.arange(t_len, dtype=torch.float32)
    angle = torch.outer(t, torch.cat([freqs, freqs]))
    trig = angle.cos() if kind == "rotary_cos" else angle.sin()
    return trig * torch.tensor(amplitude, dtype=torch.float64).to(torch.float32)


def _table_tolerance(t_len: int, amplitude: float = 1.0) -> float:
    # The angle t·ω is computed in f32: its error grows with the position.
    return max(2e-5, 5e-7 * t_len) * max(1.0, amplitude)


def _as_table(value):
    """`[T, d]` float32 if `value` has the shape of a position table `[1, …, 1, T, d]`, d even."""
    import torch

    if not hasattr(value, "shape") or not torch.is_tensor(value) or not value.is_floating_point():
        return None
    v = value.detach().to(torch.float32)
    if v.dim() < 2 or any(s != 1 for s in v.shape[:-2]):
        return None
    t_len, d = int(v.shape[-2]), int(v.shape[-1])
    if d < 2 or d % 2 or t_len < 2:
        return None
    return v.reshape(t_len, d)


def recover_rotary(cos, sin):
    """Frequencies and amplitude of a pair of rotary tables `[T, d]`, or None.

    The cosine alone is not enough: at low frequencies, `cos(t·ω)` is exactly 1 in f32 over the
    whole sequence and says nothing about ω. The pair, however, gives the angle `atan2(sin, cos)`
    at each position; unwrapped along t, its slope is ω. Each frequency is then adjusted to within
    one unit in the last place: this is what recovers the module's exact f32 frequencies.
    """
    import math

    import torch

    t_len, d = cos.shape
    half = d // 2
    amplitude = float(cos[0, 0])
    if not (math.isfinite(amplitude) and amplitude > 0):
        return None
    if not bool((cos[0] == cos[0, 0]).all()) or bool(sin[0].abs().max() > 0):
        return None
    c = cos[:, :half].double().numpy()
    s = sin[:, :half].double().numpy()
    phase = np.unwrap(np.arctan2(s, c), axis=0)
    t = np.arange(t_len, dtype=np.float64)[:, None]
    slope = (t * phase).sum(axis=0) / float((t * t).sum())
    if not np.all(np.isfinite(slope)) or np.any(slope < -1e-6):
        return None
    base = torch.from_numpy(np.maximum(slope, 0.0)).to(torch.float32)
    amp = torch.tensor(amplitude, dtype=torch.float32)
    tt = torch.arange(t_len, dtype=torch.float32)[:, None]
    candidates = [base]
    for direction in (float("inf"), float("-inf")):
        step = base
        for _ in range(2):
            step = torch.nextafter(step, torch.full_like(step, direction))
            candidates.append(step)
    best, best_err = base, None
    for cand in candidates:
        cand = torch.clamp(cand, min=0.0)
        angle = tt * cand
        err = torch.maximum(
            (angle.cos() * amp - cos[:, :half]).abs().amax(dim=0),
            (angle.sin() * amp - sin[:, :half]).abs().amax(dim=0),
        )
        if best_err is None:
            best, best_err = cand, err
        else:
            better = err < best_err
            best = torch.where(better, cand, best)
            best_err = torch.where(better, err, best_err)
    return [float(w) for w in best.tolist()], amplitude


def recognise_rotary_pair(cos_value, sin_value):
    """Frequency law of a (cosine, sine) pair of rotary tables, up to f32 rounding: `{"base": b}`
    if it is the original law, otherwise `{"frequencies": [...], "amplitude": A}` — the form of
    any scaling (linear, NTK, YaRN, LongRoPE, Llama 3.1). None if neither fits."""
    cos, sin = _as_table(cos_value), _as_table(sin_value)
    if cos is None or sin is None or cos.shape != sin.shape:
        return None
    t_len, d = int(cos.shape[0]), int(cos.shape[1])
    found = recover_rotary(cos, sin)
    if found is None:
        return None
    frequencies, amplitude = found
    tol = _table_tolerance(t_len, amplitude)

    def fits(make) -> bool:
        return all(
            float((make(kind) - tab).abs().max()) <= tol
            for kind, tab in (("rotary_cos", cos), ("rotary_sin", sin))
        )

    # The original law first: compact and readable. ω_1 = base^(−2/d).
    if amplitude == 1.0 and d >= 4 and frequencies[1] > 0:
        for base in _snap(frequencies[1] ** (-d / 2)) + [10000.0, 500000.0, 1000000.0]:
            if fits(lambda kind: position_table(kind, base, t_len, d)):
                return {"base": float(base)}
    if fits(lambda kind: rotary_table(kind, frequencies, amplitude, t_len)):
        return {"frequencies": frequencies, "amplitude": amplitude}
    return None


def freeze_rotary_embeddings(model, seq_len: int) -> List[str]:
    """Freeze the rotary modules that recompute their frequencies from the sequence length —
    dynamic NTK and LongRoPE in Hugging Face — and return their names.

    In their forward pass, these modules decide whether to switch frequencies by comparing the
    length with their original context. `torch.export` rejects such a choice on a computed value,
    before the platform even sees the model. For a fixed length, however, the choice is known:
    each module is run once at `seq_len`, in ordinary Python, which installs the frequencies for
    that length, and its forward pass is then replaced by the same one without the update. The
    model's output at that length does not change; the description captured afterwards carries
    those frequencies (ADR-0022).

    A module is affected if it has `inv_freq` and `rope_type` and its forward pass is decorated
    (the `transformers` convention).
    """
    import inspect
    import types

    import torch

    if int(seq_len) < 1:
        raise ExportError("seq_len must be a strictly positive integer")
    frozen = []
    for name, module in model.named_modules():
        forward = type(module).forward
        if not (hasattr(module, "inv_freq") and hasattr(module, "rope_type")):
            continue
        plain = inspect.unwrap(forward)
        if plain is forward:
            continue
        if not isinstance(module.rope_type, str):
            raise ExportError(
                f"rotary module '{name}' has several layer types: freeze each type by hand"
            )
        device = module.inv_freq.device
        with torch.no_grad():
            module(torch.zeros(1, int(seq_len), 1, device=device),
                   torch.arange(int(seq_len), device=device)[None])
        module.forward = types.MethodType(plain, module)
        frozen.append(name or type(module).__name__)
    return frozen


def batched_experts(model) -> List[str]:
    """Switch the mixtures of experts of a `transformers` model to the `batched_mm` implementation
    and return the names of the modules switched.

    By default, `transformers` runs the experts on the tokens each one receives: a loop whose
    length and shapes depend on the routing (`nonzero`, grouped products), which a static graph
    cannot express. `batched_mm` computes the same function with static shapes: every token is sent
    to its `k` experts by gathering their weights (ADR-0029). Call it before `export_model`.
    """
    switched = []
    for name, module in model.named_modules():
        config = getattr(module, "config", None)
        if hasattr(module, "set_experts_implementation") and hasattr(config, "_experts_implementation"):
            if config._experts_implementation != "batched_mm":
                module.set_experts_implementation("batched_mm")
                switched.append(name or type(module).__name__)
            break
    return switched


class ExportError(RuntimeError):
    """The capture failed for a stated reason."""


class UnsupportedOperator(ExportError):
    """An operator of the module is not in the published profile."""

    def __init__(self, op: str, node: str, hint: str = ""):
        msg = f"operator '{op}' (node '{node}') is not part of profile {PROFILE_VERSION}"
        if not hint and "grouped_mm" in op:
            hint = ("the experts of a transformers mixture run on data-dependent token groups: call "
                    "gpugoblin.batched_experts(model) first (ADR-0029)")
        if hint:
            msg += f"; {hint}"
        msg += f". Profile operators: {', '.join(sorted(PROFILE_ONNX_EQUIVALENTS))}"
        super().__init__(msg)
        self.op = op
        self.node = node


_NAME_RE = re.compile(r"[^A-Za-z0-9_\-./]")


def _clean_name(name: str, taken: set) -> str:
    base = _NAME_RE.sub("_", name)[:60] or "t"
    out = base
    i = 1
    while out in taken:
        i += 1
        out = f"{base}_{i}"
    taken.add(out)
    return out


def _target_name(target) -> str:
    """`aten.linear.default` -> 'aten.linear.default' (or the Python name of the function)."""
    name = getattr(target, "_schema", None)
    if name is not None:
        return str(target).replace("torch.ops.", "")
    return getattr(target, "__name__", str(target))


class _Builder:
    def __init__(self, batch: int):
        self.batch = batch
        self.nodes: List[Dict[str, Any]] = []
        self.taken: set = set()
        self.shapes: Dict[str, Tuple[int, ...]] = {}

    def emit(self, op: str, inputs: Sequence[str], out_hint: str, attrs: Optional[Dict[str, Any]] = None,
             shape: Optional[Tuple[int, ...]] = None) -> str:
        out = _clean_name(out_hint, self.taken)
        node = {"op": op, "inputs": list(inputs), "output": out}
        if attrs:
            node["attrs"] = attrs
        self.nodes.append(node)
        if shape is not None:
            self.shapes[out] = tuple(int(d) for d in shape)
        return out


def export_model(
    model,
    example_inputs,
    loss: str,
    *,
    name: str = "model",
    target_dtype: Optional[str] = None,
    target_shape: Optional[Sequence[int]] = None,
    input_names: Optional[Sequence[str]] = None,
    input_dtypes: Optional[Sequence[str]] = None,
    _missing: Optional[List[Tuple[str, str]]] = None,
):
    """Capture `model` on `example_inputs` and return `(description, weights)`.

    - `loss`: `"cross_entropy"` (i64 index target, one class per example or per token) or `"mse"`
      (target with the shape of the output).
    - `example_inputs`: a tensor or a tuple of tensors; their first dimension is the batch.
    - `input_dtypes`: `"f32"` (default) or `"i64"` for indices (embeddings).

    The returned weights are a dict `name -> np.ndarray` in PyTorch layout (`Linear` matrices stay
    `[out, in]`, the description transposes them): `load_state_dict` takes a checkpoint as is.

    The description must hold at **any** number of examples, not only at the capture batch:
    `torch.export` writes shapes as numbers, and a reshape to `[tokens·k, …]` carries the capture's
    count. The module is therefore captured twice, at the given batch and at one more example, and
    the two descriptions are compared: a dimension that follows the number of examples becomes the
    free leading dimension (-1) of its `Reshape`; anything else that differs is refused, by name
    (a description that would only run at the capture batch).
    """
    # The capture puts the module in evaluation mode (no dropout draw, no running-statistics
    # update); it is given back in the mode it came in. Left in evaluation mode, a module the
    # researcher goes on training in PyTorch would normalize with running statistics unknowingly.
    modes = [(m, m.training) for m in model.modules()] if hasattr(model, "modules") else []
    try:
        return _export_model(model, example_inputs, loss, name, target_dtype, target_shape,
                             input_names, input_dtypes, _missing)
    finally:
        for m, training in modes:
            m.training = training


def _export_model(model, example_inputs, loss, name, target_dtype, target_shape, input_names,
                  input_dtypes, _missing):
    """`export_model`, without restoring the module's mode."""
    kwargs = dict(name=name, target_dtype=target_dtype, target_shape=target_shape,
                  input_names=input_names, input_dtypes=input_dtypes)
    if _missing is not None:
        return _capture(model, example_inputs, loss, _missing=_missing, **kwargs)
    import torch

    inputs = tuple(example_inputs) if isinstance(example_inputs, (tuple, list)) else (example_inputs,)
    description, weights = _capture(model, inputs, loss, **kwargs)
    if not inputs or any(t.ndim == 0 for t in inputs):
        return description, weights
    more = tuple(torch.cat([t, t[:1]]) for t in inputs)
    other, _ = _capture(model, more, loss, **kwargs)
    return _merge_batch(description, other, int(inputs[0].shape[0]), int(more[0].shape[0])), weights


def _merge_batch(d1: Dict[str, Any], d2: Dict[str, Any], b1: int, b2: int) -> Dict[str, Any]:
    """One description valid at every batch size, from two captures at `b1` and `b2` examples."""
    def refuse(what: str):
        raise ExportError(
            f"{what} depends on the number of examples: the description would only run at the "
            f"capture batch ({b1}). Capture with a batch of at least 2 if it is 1; otherwise this is "
            "a module or driver construct that ties a shape to the batch (please report it)"
        )

    for key in ("format", "name", "inputs", "parameters", "loss"):
        if d1.get(key) != d2.get(key):
            refuse(f"the '{key}' section")
    n1, n2 = d1["nodes"], d2["nodes"]
    if len(n1) != len(n2) or any((a["op"], a["inputs"], a["output"]) != (b["op"], b["inputs"], b["output"])
                                 for a, b in zip(n1, n2)):
        refuse("the structure of the captured graph")
    out = dict(d1)
    nodes = []
    for a, b in zip(n1, n2):
        attrs_a, attrs_b = a.get("attrs", {}), b.get("attrs", {})
        if attrs_a == attrs_b:
            nodes.append(a)
            continue
        merged = dict(attrs_a)
        for k in set(attrs_a) | set(attrs_b):
            va, vb = attrs_a.get(k), attrs_b.get(k)
            if va == vb:
                continue
            if a["op"] != "Reshape" or k != "shape" or len(va) != len(vb):
                refuse(f"node '{a['output']}' ({a['op']}): attribute '{k}' ({va} at {b1}, {vb} at {b2})")
            shape = []
            for x, y in zip(va, vb):
                if x == y:
                    shape.append(x)
                elif x == -1 or y == -1:
                    # One capture read a constant as the batch because they coincided.
                    shape.append(y if x == -1 else x)
                elif x > 0 and y > 0 and x * b2 == y * b1:
                    shape.append(-1)
                else:
                    refuse(f"node '{a['output']}' (Reshape): dimension {x} at {b1}, {y} at {b2}")
            free = [i for i, v in enumerate(shape) if v == -1]
            if len(free) > 1 or (free and free[0] != 0):
                refuse(f"node '{a['output']}' (Reshape): shape {shape}, the batch not in the leading position")
            merged[k] = shape
        nodes.append(dict(a, attrs=merged))
    out["nodes"] = nodes
    return out


def _capture(
    model,
    example_inputs,
    loss: str,
    *,
    name: str = "model",
    target_dtype: Optional[str] = None,
    target_shape: Optional[Sequence[int]] = None,
    input_names: Optional[Sequence[str]] = None,
    input_dtypes: Optional[Sequence[str]] = None,
    _missing: Optional[List[Tuple[str, str]]] = None,
):
    """One capture at the batch of `example_inputs` (see `export_model`)."""
    try:
        import torch
        from torch.export import export as torch_export
    except ImportError as e:  # pragma: no cover
        raise ExportError("torch is required to capture a module") from e

    if loss not in ("cross_entropy", "mse"):
        raise ExportError("loss must be 'cross_entropy' or 'mse'")
    if not isinstance(example_inputs, (tuple, list)):
        example_inputs = (example_inputs,)
    example_inputs = tuple(example_inputs)
    if not example_inputs or any(t.ndim == 0 for t in example_inputs):
        raise ExportError("at least one example input of rank ≥ 1 is required")
    batch = int(example_inputs[0].shape[0])
    for t in example_inputs:
        if int(t.shape[0]) != batch:
            raise ExportError("all inputs must have the same batch dimension first")

    model = model.eval() if hasattr(model, "eval") else model
    # A parameter frozen on the PyTorch side (`requires_grad=False`, an adapter's base) stays frozen.
    frozen_keys = (
        {n for n, p in model.named_parameters() if not p.requires_grad}
        if hasattr(model, "named_parameters") else set()
    )
    try:
        ep = torch_export(model, example_inputs)
    except Exception as e:
        if type(e).__name__ != "GuardOnDataDependentSymNode":
            raise
        raise ExportError(
            "torch.export rejects a choice that depends on a value computed in the module. If it is "
            "a RoPE that switches frequencies with the length (dynamic NTK, Hugging Face LongRoPE), "
            "call freeze_rotary_embeddings(module, length) first; otherwise, make this choice "
            f"independent of the data. Detail: {str(e).splitlines()[0]}"
        ) from e
    gm = ep.graph_module
    sig = ep.graph_signature

    # Placeholders: parameters, buffers, user inputs.
    param_of = dict(sig.inputs_to_parameters)
    buffer_of = dict(sig.inputs_to_buffers)
    user_inputs = list(sig.user_inputs)
    state = {**{k: v.detach().cpu().numpy() for k, v in ep.state_dict.items()}}
    constants = {k: v for k, v in getattr(ep, "constants", {}).items()}

    values = _node_values(ep, gm, example_inputs)
    const_nodes = _constant_cone(gm, set(param_of), set(user_inputs))
    # Nodes that depend on an input: only they carry the batch axis first. A parameter, or a
    # tensor computed from parameters alone (a relative position table), has no batch axis.
    batch_nodes = _input_cone(gm, set(user_inputs))

    b = _Builder(batch)
    env: Dict[Any, str] = {}          # FX node -> gg tensor name
    param_names: Dict[str, str] = {}  # state_dict key -> gg name
    parameters: List[Dict[str, Any]] = []
    weights: Dict[str, np.ndarray] = {}
    inputs_spec: List[Dict[str, Any]] = []
    names_in = list(input_names) if input_names else [f"x{i}" if i else "x" for i in range(len(user_inputs))]
    dtypes_in = list(input_dtypes) if input_dtypes else ["f32"] * len(user_inputs)
    if len(names_in) != len(user_inputs) or len(dtypes_in) != len(user_inputs):
        raise ExportError(f"the module takes {len(user_inputs)} input(s)")

    def add_param(key: str, arr) -> str:
        if key in param_names:
            return param_names[key]
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 0 or a.ndim > 4:
            raise ExportError(f"parameter '{key}' has rank {a.ndim}, the profile accepts 1 to 4")
        gg = _clean_name(key, b.taken)
        param_names[key] = gg
        spec = {"name": gg, "shape": [int(d) for d in a.shape], "init": "zeros"}
        if key in frozen_keys:
            spec["trainable"] = False
        parameters.append(spec)
        weights[gg] = a
        b.shapes[gg] = tuple(int(d) for d in a.shape)
        return gg

    tied = tied_parameters(model)
    ui = 0
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        if node.name in param_of:
            # A tied parameter (the language-model head sharing the embedding) is lifted by
            # torch.export once per alias: every alias becomes the same gg parameter, so its
            # gradient sums the uses, as in PyTorch.
            key = tied.get(param_of[node.name], param_of[node.name])
            env[node] = add_param(key, state[key])
        elif node.name in buffer_of:
            key = buffer_of[node.name]
            env[node] = ("buffer", key)
        elif node.name in user_inputs:
            t = example_inputs[ui]
            gg = _clean_name(names_in[ui], b.taken)
            shape = [0] + [int(d) for d in t.shape[1:]]
            inputs_spec.append({"name": gg, "shape": shape, "dtype": dtypes_in[ui]})
            b.shapes[gg] = tuple([batch] + [int(d) for d in t.shape[1:]])
            env[node] = gg
            ui += 1
        else:
            env[node] = ("constant", node.name)

    constants_by_value: Dict[float, str] = {}

    def const_param(value: float, ctx: str) -> str:
        """A scalar of the model, declared as a frozen parameter with a constant initializer."""
        value = float(value)
        if not np.isfinite(value):
            raise UnsupportedOperator("non-finite constant", ctx, f"{value!r}")
        if value in constants_by_value:
            return constants_by_value[value]
        gg = _clean_name(f"const.{len(constants_by_value)}", b.taken)
        parameters.append({
            "name": gg, "shape": [1],
            "init": {"constant": {"value": value}},
            "trainable": False,
        })
        weights[gg] = np.array([value], dtype=np.float32)
        b.shapes[gg] = (1,)
        constants_by_value[value] = gg
        return gg

    tables_by_node: Dict[Any, str] = {}
    inits_by_node: Dict[Any, Dict[str, Any]] = {}

    def rotary_partner(node):
        """Shared law of the rotary table `node` and its partner (the sine of a cosine, or the
        reverse), searched among the constant tensors of the same shape. Both tables receive the
        same law: their descriptions cannot diverge."""
        v = values.get(node)
        mine = _as_table(v)
        if mine is None:
            return None
        cosine = bool((mine[0] == mine[0, 0]).all()) and float(mine[0, 0]) > 0
        sine = bool((mine[0] == 0).all())
        if not (cosine or sine):
            return None
        for other in gm.graph.nodes:
            if other is node or other not in const_nodes:
                continue
            w = values.get(other)
            if not hasattr(w, "shape") or tuple(w.shape) != tuple(v.shape):
                continue
            pair = (node, other) if cosine else (other, node)
            law = recognise_rotary_pair(values[pair[0]], values[pair[1]])
            if law is not None:
                inits_by_node[pair[0]] = {"rotary_cos": law}
                inits_by_node[pair[1]] = {"rotary_sin": law}
                return inits_by_node[node]
        return None

    def table_param(node, ctx: str) -> Optional[str]:
        """A constant position table of the module, turned into a frozen parameter with a named
        generator. The checkpoint carries the module's exact values; the description, their
        generator."""
        if node in tables_by_node:
            return tables_by_node[node]
        v = values.get(node)
        if v is None or not hasattr(v, "shape"):
            return None
        init = inits_by_node.get(node) or rotary_partner(node) or recognise_position_table(v)
        if init is None:
            return None
        kind = next(iter(init))
        gg = _clean_name(f"pos.{kind}.{len(tables_by_node)}", b.taken)
        arr = v.detach().to(dtype=__import__("torch").float32).cpu().numpy()
        parameters.append({"name": gg, "shape": [int(s) for s in arr.shape], "init": init, "trainable": False})
        weights[gg] = arr
        b.shapes[gg] = tuple(int(s) for s in arr.shape)
        tables_by_node[node] = gg
        return gg

    def uniform_value(node) -> Optional[float]:
        """Value of a constant node whose elements are all equal, otherwise None."""
        v = values.get(node)
        if v is None or not hasattr(v, "flatten"):
            return None
        flat = v.detach().flatten().float()
        if flat.numel() == 0 or not bool((flat == flat[0]).all()):
            return None
        return float(flat[0])

    def tensor_of(arg, ctx: str) -> str:
        if isinstance(arg, (int, float)) and not isinstance(arg, bool):
            return const_param(arg, ctx)
        if arg in const_nodes:
            u = uniform_value(arg)
            if u is not None:
                return const_param(u, ctx)
            table = table_param(arg, ctx)
            if table is not None:
                return table
            raise UnsupportedOperator(
                "non-uniform constant", ctx,
                f"'{arg.name}' is a constant tensor of the module (buffer or computed value): "
                "only scalars, the causal mask and sinusoidal or rotary position tables (scaled "
                "ones included, recognized as a cosine/sine pair) are translated",
            )
        v = env.get(arg) if not isinstance(arg, (list, tuple)) else None
        if isinstance(v, str):
            return v
        if isinstance(v, tuple) and v[0] == "buffer":
            raise UnsupportedOperator("buffer", ctx, f"buffer '{v[1]}' is neither a parameter nor an input")
        if isinstance(v, tuple) and v[0] == "constant":
            raise UnsupportedOperator("constant", ctx, "constants are not part of gg-graph/v1")
        if isinstance(v, tuple) and v[0] == "integer":
            raise UnsupportedOperator(v[1], ctx, v[2])
        raise UnsupportedOperator("scalar argument", ctx, f"{arg!r} is not a tensor")

    def shape_of(gg: str) -> Tuple[int, ...]:
        return b.shapes.get(gg, ())

    def broadcast_shape(*shapes) -> Tuple[int, ...]:
        """Broadcast shape, right-aligned as in PyTorch: [1, 64, 1, 4] and [N, 64, 8, 1] give
        [N, 64, 8, 4]."""
        rank = max(len(sh) for sh in shapes)
        out = []
        for k in range(rank):
            dims = [sh[len(sh) - rank + k] for sh in shapes if len(sh) - rank + k >= 0]
            out.append(max(dims))
        return tuple(out)

    def causal_from(x: str, mask_node, fill, masked_where_true: bool, ctx: str) -> str:
        """`CausalMask(x)` if the constant mask is exactly the future positions."""
        if fill is None or not (float(fill) <= MASK_FILL_THRESHOLD):
            raise UnsupportedOperator(
                "masked_fill / where", ctx,
                f'fill value {fill!r}: only a "−∞" mask (≤ {MASK_FILL_THRESHOLD:g}) '
                "before Softmax is translated",
            )
        if mask_node not in const_nodes:
            raise UnsupportedOperator(
                "masked_fill / where", ctx,
                "the mask depends on the data; only a constant causal mask is translated",
            )
        m = values[mask_node]
        if not masked_where_true:
            m = ~m.bool()
        sx = shape_of(x)
        if len(sx) < 3 or sx[-1] != sx[-2]:
            raise UnsupportedOperator("masked_fill / where", ctx, f"scores of shape {sx}, expected [batch, …, T, T]")
        n = sx[-1]
        import torch

        future = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        m = m.bool()
        if m.dim() < 2 or tuple(m.shape[-2:]) != (n, n) or not bool((m == future).all()):
            raise UnsupportedOperator(
                "masked_fill / where", ctx,
                "the mask is not the strict triangle of future positions; only the causal mask "
                "is translated",
            )
        return b.emit("CausalMask", [x], ctx, shape=sx)

    def maybe(arg):
        return None if arg is None else tensor_of(arg, "?")

    def as_pair(v) -> list:
        if isinstance(v, (list, tuple)):
            v = list(v)
            return [int(v[0]), int(v[-1])]
        return [int(v), int(v)]

    def batch_reshape(dims: Sequence[int], input_shape: Tuple[int, ...]) -> list:
        out = [int(d) for d in dims]
        if out and input_shape and out[0] == batch:
            out[0] = -1
        return out

    def concat_copies(x: str, axis: int, n: int, ctx: str) -> str:
        """`n` copies of `x` placed end to end along `axis`, in groups of at most eight (the limit of
        `Concat`)."""
        parts = [x] * n
        level = 0
        while len(parts) > 1:
            grouped = []
            for i in range(0, len(parts), 8):
                chunk = parts[i:i + 8]
                if len(chunk) == 1:
                    grouped.append(chunk[0])
                    continue
                s = list(shape_of(chunk[0]))
                s[axis] = sum(shape_of(c)[axis] for c in chunk)
                grouped.append(b.emit("Concat", chunk, f"{ctx}_rep{level}_{i}", {"axis": axis}, tuple(s)))
            parts = grouped
            level += 1
        return parts[0]

    def repeat_heads(x: str, axis: int, n: int, ctx: str) -> str:
        """Repeat each slice of `x` along `axis` `n` times in a row (`repeat_kv`,
        `repeat_interleave`) without a rank-5 tensor: `n` copies placed end to end along the next
        axis, then a `Reshape` that folds that axis into `axis`. This is what translates the
        grouped-query attention (GQA) of Llama 3, Qwen 2 and Mistral."""
        sx = shape_of(x)
        if n == 1:
            return x
        if axis + 1 >= len(sx) or axis == 0:
            raise UnsupportedOperator("head repetition", ctx, f"axis {axis} of {sx}")
        cat = concat_copies(x, axis + 1, n, ctx)
        target = list(sx)
        target[axis] = sx[axis] * n
        target[axis + 1] = sx[axis + 1]
        dims = batch_reshape(target, sx)
        return b.emit("Reshape", [cat], f"{ctx}_heads", {"shape": dims}, tuple(target))

    def mask_is_empty(mask_node) -> bool:
        """True for a constant mask that masks nothing: boolean all true, or additive all zero."""
        if mask_node not in const_nodes:
            return False
        m = values[mask_node]
        return bool(m.all()) if m.dtype == __import__("torch").bool else bool((m == 0).all())

    def attention_mask_is_causal(mask_node, n: int, ctx: str) -> bool:
        """A constant `attn_mask` that is exactly the causal mask: boolean true on and below the
        diagonal, or additive zero on and below the diagonal and "−∞" above it."""
        import torch

        if mask_node not in const_nodes:
            raise UnsupportedOperator("attn_mask", ctx, "the mask depends on the data; only a constant causal mask is translated")
        m = values[mask_node]
        if m.dim() < 2 or tuple(m.shape[-2:]) != (n, n):
            raise UnsupportedOperator("attn_mask", ctx, f"mask of shape {tuple(m.shape)}")
        keep = torch.tril(torch.ones(n, n, dtype=torch.bool)).expand_as(m)
        if m.dtype == torch.bool:
            ok = bool((m == keep).all())
        else:
            ok = bool((m.masked_select(keep) == 0).all()) and bool(
                (m.masked_select(~keep) <= MASK_FILL_THRESHOLD).all()
            )
        if not ok:
            raise UnsupportedOperator("attn_mask", ctx, "the mask is not the causal mask; only that one is translated")
        return True

    def formula(expr: str, ins: Sequence[str], ctx: str) -> str:
        """A `Formula` node: `expr` over the broadcast inputs `a`, `b`, … (ADR-0025)."""
        return b.emit("Formula", list(ins), ctx, {"expr": expr}, broadcast_shape(*[shape_of(i) for i in ins]))

    def reshape(x: str, target: Sequence[int], ctx: str) -> str:
        target = [int(d) for d in target]
        if tuple(target) == tuple(shape_of(x)):
            return x
        return b.emit("Reshape", [x], ctx, {"shape": batch_reshape(target, shape_of(x))}, tuple(target))

    def slice_(x: str, axis: int, start: int, end: int, ctx: str):
        sx = shape_of(x)
        if start == 0 and end == sx[axis]:
            return x
        if end <= start:
            # Empty slice (partial RoPE whose non-rotated part is empty): nothing to emit; only a
            # concatenation can consume it, and it ignores it.
            return ("empty", axis)
        if axis == 0 and x not in {p["name"] for p in parameters}:
            raise UnsupportedOperator("slice", ctx, "slice along the batch axis")
        out = list(sx)
        out[axis] = end - start
        return b.emit("Slice", [x], ctx, {"axis": axis, "start": start, "end": end}, tuple(out))

    def concat_tree(parts: List[str], axis: int, ctx: str) -> str:
        """Concatenation of `parts` along `axis`, in groups of at most eight (the limit of `Concat`)."""
        level = 0
        while len(parts) > 1:
            grouped = []
            for i in range(0, len(parts), 8):
                chunk = parts[i:i + 8]
                if len(chunk) == 1:
                    grouped.append(chunk[0])
                    continue
                sh = list(shape_of(chunk[0]))
                sh[axis] = sum(shape_of(c)[axis] for c in chunk)
                name = ctx if len(parts) <= 8 else f"{ctx}_cat{level}_{i}"
                grouped.append(b.emit("Concat", chunk, name, {"axis": axis}, tuple(sh)))
            parts = grouped
            level += 1
        return parts[0]

    batch_refs: Dict[int, str] = {}

    def batch_ref(rank: int, ctx: str) -> str:
        """A tensor `[batch, 1, …, 1]` of rank `rank`, taken from the first float input: what carries
        the batch dimension in a broadcast (`expand` of a learned token over the batch)."""
        if rank in batch_refs:
            return batch_refs[rank]
        src = next((i["name"] for i in inputs_spec if i["dtype"] == "f32"), None)
        if src is None:
            raise UnsupportedOperator("expand", ctx, "broadcast over the batch without a float input")
        flat = reshape(src, (batch, int(np.prod(shape_of(src)[1:]))), f"{ctx}_batch")
        one = slice_(flat, 1, 0, 1, f"{ctx}_batch1")
        batch_refs[rank] = reshape(one, (batch,) + (1,) * (rank - 1), f"{ctx}_batch{rank}")
        return batch_refs[rank]

    def zeros_param(shape: Sequence[int], ctx: str) -> str:
        gg = _clean_name(f"zeros.{ctx}", b.taken)
        parameters.append({"name": gg, "shape": [int(d) for d in shape], "init": "zeros", "trainable": False})
        weights[gg] = np.zeros([int(d) for d in shape], dtype=np.float32)
        b.shapes[gg] = tuple(int(d) for d in shape)
        return gg

    def dim_arg(v, rank: int) -> int:
        return int(v) % rank

    # ---- gg-ops/7 (ADR-0027) ------------------------------------------------------------------

    def lit(v) -> str:
        """A Python number as a literal of the Formula grammar."""
        v = float(v)
        if not np.isfinite(v):
            raise ExportError(f"non-finite constant {v!r}")
        return f"({v!r})" if v < 0 or "e" in repr(v) else repr(v)

    def is_number(v) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    def formula_of_args(template: str, args, ctx: str) -> str:
        """`template` with `{a}`, `{b}`, `{c}` replaced by the inputs in order, or by literals for
        the Python numbers among them."""
        ins, names = [], {}
        for key, arg in zip("abc", args):
            if is_number(arg):
                names[key] = lit(arg)
            else:
                names[key] = "abcd"[len(ins)]
                ins.append(tensor_of(arg, ctx))
        if not ins:
            raise ExportError(f"node '{ctx}': no tensor among the arguments")
        return formula(template.format(**names), ins, ctx)

    def has_batch(arg) -> bool:
        return arg in batch_nodes

    def axis_arg(arg, v, ctx: str, what: str) -> int:
        """A non-negative axis of `arg`, refusing the batch axis of a tensor that carries it."""
        rank = len(shape_of(tensor_of(arg, ctx)))
        d = int(v) % rank
        if d == 0 and has_batch(arg):
            raise UnsupportedOperator(what, ctx, "along the batch axis, which would mix examples")
        return d

    def reduce_chain(op_name: str, x: str, axes, keep: bool, ctx: str) -> str:
        """A reduction over several axes, one node per axis (the last ones first)."""
        out, cur = list(shape_of(x)), x
        axes = sorted(set(axes), reverse=True)
        for k, d in enumerate(axes):
            if keep:
                out[d] = 1
            else:
                out.pop(d)
            hint = ctx if k == len(axes) - 1 else f"{ctx}_{d}"
            cur = b.emit(op_name, [cur], hint, {"axis": d, "keepdims": 1 if keep else 0}, tuple(out))
        return cur

    def reduce_axes(arg, dims, ctx: str, what: str) -> List[int]:
        x = tensor_of(arg, ctx)
        rank = len(shape_of(x))
        if dims is None or (isinstance(dims, (list, tuple)) and len(dims) == 0):
            dims = list(range(rank))
        dims = [dims] if isinstance(dims, int) else list(dims)
        axes = sorted({int(d) % rank for d in dims})
        if 0 in axes and has_batch(arg):
            raise UnsupportedOperator(what, ctx, "a reduction over the batch axis would mix examples")
        return axes

    def integer_result(what: str, ctx: str) -> Tuple[str, str, str]:
        return ("integer", what, f"{ctx}: integer indices ({what}) are not tensors of gg-graph/v1, "
                "which carries f32 between nodes (ADR-0027)")

    def const_index(arg, ctx: str, what: str):
        """The value of a constant index tensor (a buffer, or computed from constants only)."""
        import torch

        if arg not in const_nodes:
            raise UnsupportedOperator(
                what, ctx, "the index depends on the data; only constant indices (a buffer, a fixed "
                "graph) are translated: an index computed in the graph needs integer tensors "
                "(ADR-0027)")
        v = values[arg]
        if not torch.is_tensor(v):
            v = torch.as_tensor(v)
        if v.dtype == torch.bool:
            if v.dim() != 1:
                raise UnsupportedOperator(what, ctx, "boolean mask of rank > 1")
            v = v.nonzero().flatten()
        return v.to(torch.int64)

    def index_select(x: str, axis: int, index, ctx: str) -> str:
        """`IndexSelect` of a 1-D constant index; negative indices counted from the end."""
        sx = shape_of(x)
        idx = [int(i) % sx[axis] if int(i) < 0 else int(i) for i in index]
        if not idx:
            raise UnsupportedOperator("index", ctx, "empty index")
        if idx == list(range(sx[axis])):
            return x
        out = list(sx)
        out[axis] = len(idx)
        return b.emit("IndexSelect", [x], ctx, {"axis": axis, "index": idx}, tuple(out))

    def pad_node(x: str, pairs, mode: str, value, ctx: str, arg) -> str:
        """`Pad` from per-axis `(begin, end)` pairs; a negative padding is a crop (`Slice`)."""
        sx = list(shape_of(x))
        for axis, (p0, p1) in enumerate(pairs):
            if (p0 or p1) and axis == 0 and has_batch(arg):
                raise UnsupportedOperator("pad", ctx, "padding along the batch axis")
            if p0 < 0 or p1 < 0:
                if mode != "constant":
                    raise UnsupportedOperator("pad", ctx, f"negative {mode} padding")
                x = slice_(x, axis, -min(p0, 0), sx[axis] + min(p1, 0), f"{ctx}_crop{axis}")
                sx = list(shape_of(x))
        pairs = [(max(p0, 0), max(p1, 0)) for p0, p1 in pairs]
        if all(p == (0, 0) for p in pairs):
            return x
        pads = [p[0] for p in pairs] + [p[1] for p in pairs]
        attrs: Dict[str, Any] = {"pads": pads}
        if mode != "constant":
            attrs["mode"] = mode
        elif value:
            attrs["value"] = float(value)
        out = tuple(d + p0 + p1 for d, (p0, p1) in zip(sx, pairs))
        return b.emit("Pad", [x], ctx, attrs, out)

    def torch_pad_pairs(rank: int, pad) -> List[Tuple[int, int]]:
        """PyTorch's padding (last axis first: `[l, r, t, b, …]`) as per-axis pairs."""
        pad = [int(v) for v in pad]
        pairs = [(0, 0)] * rank
        for k in range(len(pad) // 2):
            pairs[rank - 1 - k] = (pad[2 * k], pad[2 * k + 1])
        return pairs

    def circular_pad(x: str, pairs, ctx: str) -> str:
        for axis, (p0, p1) in enumerate(pairs):
            if not (p0 or p1):
                continue
            n = shape_of(x)[axis]
            if p0 > n or p1 > n or p0 < 0 or p1 < 0:
                raise UnsupportedOperator("circular pad", ctx, f"padding ({p0}, {p1}) of an axis of {n}")
            parts = []
            if p0:
                parts.append(slice_(x, axis, n - p0, n, f"{ctx}_l{axis}"))
            parts.append(x)
            if p1:
                parts.append(slice_(x, axis, 0, p1, f"{ctx}_r{axis}"))
            x = concat_tree(parts, axis, f"{ctx}_c{axis}")
        return x

    def resize(node, t: str, a, kw, mode: str) -> str:
        """`upsample_*` as `Resize`, after checking that PyTorch's source indices are the ones the
        executor computes from the sizes alone (they differ when a non-integer scale factor is
        passed instead of a size)."""
        name = node.name
        x = tensor_of(a[0], name)
        sx = shape_of(x)
        spatial = len(sx) - 2
        align = False
        if mode == "linear":
            align = bool(a[2]) if len(a) > 2 else bool(kw.get("align_corners", False))
        out_size = list(node.meta["val"].shape[2:])
        # Scale factors passed by PyTorch, if any: `.vec` (…, output_size, [align,] scales) or the
        # explicit form (…, output_size, [align,] scales_h, scales_w).
        scales: List[Optional[float]] = [None] * spatial
        if t.endswith(".vec"):
            sf = a[-1] if len(a) > (3 if mode == "linear" else 2) else kw.get("scale_factors")
            if sf is not None:
                scales = [float(v) for v in sf]
        else:
            first = 3 if mode == "linear" else 2
            tail = list(a[first:first + spatial])
            for k in range(spatial):
                v = tail[k] if k < len(tail) else kw.get(("scales_h", "scales_w")[k] if spatial == 2 else "scales")
                scales[k] = None if v is None else float(v)
        f32 = np.float32
        for k in range(spatial):
            inp, out = int(sx[2 + k]), int(out_size[k])
            s = scales[k]
            if s is None or (mode == "linear" and align):
                continue
            theirs = f32(1.0 / s) if s > 0 else f32(inp) / f32(out)
            ours = f32(inp) / f32(out)
            o = np.arange(out, dtype=f32)
            if mode == "nearest":
                same = np.array_equal(np.minimum(np.floor(o * theirs), inp - 1), np.minimum(np.floor(o * ours), inp - 1))
            else:
                src_t = np.maximum(theirs * (o + f32(0.5)) - f32(0.5), 0)
                src_o = np.maximum(ours * (o + f32(0.5)) - f32(0.5), 0)
                same = np.array_equal(src_t, src_o)
            if not same:
                raise UnsupportedOperator(t, name, f"scale factor {s} whose source coordinates differ from {inp} → {out}")
        attrs: Dict[str, Any] = {"size": [int(v) for v in out_size]}
        if mode == "linear":
            attrs["mode"] = "linear"
            if align:
                attrs["align_corners"] = 1
        return b.emit("Resize", [x], name, attrs, tuple(sx[:2]) + tuple(int(v) for v in out_size))

    def conv_transpose(node, t: str, a, kw) -> str:
        """1D or 2D transposed convolution; a 1D one is a `ConvTranspose2d` of height 1."""
        name = node.name
        x, w = tensor_of(a[0], name), tensor_of(a[1], name)
        sx, sw = shape_of(x), shape_of(w)
        bias = a[2] if len(a) > 2 else kw.get("bias")
        one_d = len(sw) == 3
        if len(sw) not in (3, 4):
            raise UnsupportedOperator(t, name, f"kernel of rank {len(sw)}")
        n = 1 if one_d else 2

        def tup(v, default):
            if v is None:
                return [default] * n
            v = list(v) if isinstance(v, (list, tuple)) else [v]
            return [int(v[0])] * n if len(v) == 1 else [int(e) for e in v]

        if t.startswith("aten.convolution"):
            # (input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)
            stride, padding, dilation = tup(a[3], 1), tup(a[4], 0), tup(a[5], 1)
            out_pad, groups = tup(a[7], 0), int(a[8])
        else:
            # conv_transpose{1,2}d(input, weight, bias, stride, padding, output_padding, groups, dilation)
            stride = tup(a[3] if len(a) > 3 else kw.get("stride"), 1)
            padding = tup(a[4] if len(a) > 4 else kw.get("padding"), 0)
            out_pad = tup(a[5] if len(a) > 5 else kw.get("output_padding"), 0)
            groups = int(a[6]) if len(a) > 6 else int(kw.get("groups", 1))
            dilation = tup(a[7] if len(a) > 7 else kw.get("dilation"), 1)
        if one_d:
            x = reshape(x, (sx[0], sx[1], 1, sx[2]), f"{name}_x4")
            w = reshape(w, (sw[0], sw[1], 1, sw[2]), f"{name}_w4")
            stride, padding, out_pad, dilation = [1] + stride, [0] + padding, [0] + out_pad, [1] + dilation
        ins = [x, w]
        if bias is not None:
            ins.append(tensor_of(bias, name))
        s4, w4 = shape_of(x), shape_of(w)
        dims = [
            (s4[2 + k] - 1) * stride[k] - 2 * padding[k] + dilation[k] * (w4[2 + k] - 1) + out_pad[k] + 1
            for k in range(2)
        ]
        attrs = {"stride": stride, "padding": padding, "output_padding": out_pad,
                 "dilation": dilation, "groups": groups}
        cout = w4[1] * groups
        y = b.emit("ConvTranspose2d", ins, f"{name}_2d" if one_d else name, attrs, (s4[0], cout, dims[0], dims[1]))
        return reshape(y, (s4[0], cout, dims[1]), name) if one_d else y

    def weight_norm(node, a) -> str:
        """`_weight_norm(v, g, dim)` = `v · g / ‖v‖`, the norm taken over every axis except `dim`
        (all of them for dim = −1): `ReduceSum` of the squares, then one `Formula`."""
        name = node.name
        v, g = tensor_of(a[0], name), tensor_of(a[1], name)
        sv = shape_of(v)
        dim = int(a[2]) if len(a) > 2 else 0
        axes = list(range(len(sv))) if dim == -1 else [k for k in range(len(sv)) if k != dim % len(sv)]
        sq = b.emit("Mul", [v, v], f"{name}_sq", shape=sv)
        s = reduce_chain("ReduceSum", sq, axes, True, f"{name}_ss")
        return formula("a * (b / sqrt(c))", [v, g, s], name)

    def softmax_parts(x: str, axis: int, ctx: str):
        """`x − max` and `Σ exp(x − max)` along `axis` (kept): the stable form of PyTorch."""
        sx = shape_of(x)
        red = list(sx)
        red[axis] = 1
        m = b.emit("ReduceMax", [x], f"{ctx}_max", {"axis": axis, "keepdims": 1}, tuple(red))
        shifted = b.emit("Sub", [x, m], f"{ctx}_shift", shape=sx)
        e = formula("exp(a)", [shifted], f"{ctx}_exp")
        z = b.emit("ReduceSum", [e], f"{ctx}_sum", {"axis": axis, "keepdims": 1}, tuple(red))
        return m, shifted, z

    def gather_as_select(x_arg, dim, index_arg, ctx: str, what: str):
        """The 1-D index of a constant `gather`/`scatter_add` index that only varies along `dim`
        (the broadcast of a vector), or an explicit rejection."""
        import torch

        idx = const_index(index_arg, ctx, what)
        x = tensor_of(x_arg, ctx)
        sx = shape_of(x)
        d = int(dim) % len(sx)
        if idx.dim() != len(sx):
            raise UnsupportedOperator(what, ctx, f"index of rank {idx.dim()} for a tensor of rank {len(sx)}")
        line = idx.movedim(d, -1).reshape(-1, idx.shape[d])
        if not bool((line == line[0]).all()):
            raise UnsupportedOperator(
                what, ctx, "a constant index that varies along the other axes: only the broadcast of "
                "one vector along the axis (index_select / index_add form) is translated; a description "
                "carries no constant integer tensor (an index computed in the graph is, ADR-0029)")
        other = [int(idx.shape[k]) for k in range(len(sx)) if k != d]
        return d, [int(i) for i in line[0].tolist()], other

    def translate_ops7(node, t: str, op: str, a, kw):
        """Translations added with `gg-ops/7` (ADR-0027). Returns the gg name (or an environment
        value), or None if `t` is not handled here."""
        name = node.name
        import torch

        if t.startswith("aten.expand_as"):
            # `expand_as(x, other)`: the expansion to the shape PyTorch measured (F.normalize).
            return translate_extra(node, "aten.expand.default", (a[0], list(node.meta["val"].shape)), kw)
        if t.startswith("aten.convolution") and len(a) > 6 and bool(a[6]):
            return conv_transpose(node, t, a, kw)
        if t.startswith(("aten.conv_transpose1d", "aten.conv_transpose2d")):
            return conv_transpose(node, t, a, kw)
        if t.startswith("aten._weight_norm_interface"):
            w = weight_norm(node, a)
            return ("multi", [w, integer_result("weight_norm norms", name)])
        if t.startswith("aten._weight_norm"):
            return weight_norm(node, a)
        if op == "round" and t.startswith("aten.round.decimals"):
            d = int(a[1]) if len(a) > 1 else int(kw.get("decimals", 0))
            x = tensor_of(a[0], name)
            if d >= 0:
                k = lit(10.0 ** d)
                return formula(f"round(a * {k}) / {k}", [x], name)
            k = lit(10.0 ** (-d))
            return formula(f"round(a / {k}) * {k}", [x], name)
        if op in BINARY_FORMULAS and not t.startswith("aten.__"):
            return formula_of_args(BINARY_FORMULAS[op], a[:2], name)
        if t.startswith(("aten.div.Tensor_mode", "aten.div.Scalar_mode")):
            mode = kw.get("rounding_mode", a[2] if len(a) > 2 else None)
            if mode is None:
                return formula_of_args("{a} / {b}", a[:2], name)
            fn = {"floor": "floor", "trunc": "trunc"}.get(mode)
            if fn is None:
                raise UnsupportedOperator(t, name, f"rounding mode {mode!r}")
            return formula_of_args(fn + "({a} / {b})", a[:2], name)
        if t.startswith(("aten.add.Tensor", "aten.sub.Tensor", "aten.add.Scalar", "aten.sub.Scalar")) and (
                len(a) > 2 or "alpha" in kw):
            alpha = float(a[2]) if len(a) > 2 else float(kw["alpha"])
            sign = "+" if op == "add" else "-"
            return formula_of_args(f"{{a}} {sign} {{b}} * {lit(alpha)}", a[:2], name)
        if t.startswith(("aten.clamp.Tensor", "aten.clip.Tensor")):
            lo = a[1] if len(a) > 1 else kw.get("min")
            hi = a[2] if len(a) > 2 else kw.get("max")
            args, expr = [a[0]], "{a}"
            for bound, fn in ((lo, "max"), (hi, "min")):
                if bound is not None:
                    key = "bc"[len(args) - 1]
                    args.append(bound)
                    expr = f"{fn}({expr}, {{{key}}})"
            return formula_of_args(expr, args, name)
        if t.startswith("aten.lerp.Scalar"):
            w = float(a[2])
            # PyTorch: `a + w·(b − a)` below one half, `b − (b − a)·(1 − w)` from there.
            expr = f"{{a}} + {lit(w)} * ({{b}} - {{a}})" if abs(w) < 0.5 else f"{{b}} - ({{b}} - {{a}}) * {lit(1.0 - w)}"
            return formula_of_args(expr, a[:2], name)
        if t.startswith("aten.lerp.Tensor"):
            return formula_of_args("{a} + {c} * ({b} - {a})", a[:3], name)
        if t.startswith(("aten.addcmul", "aten.addcdiv")):
            value = float(kw.get("value", a[3] if len(a) > 3 else 1.0))
            opch = "*" if "addcmul" in t else "/"
            return formula_of_args(f"{{a}} + {lit(value)} * {{b}} {opch} {{c}}", a[:3], name)
        if t.startswith("aten.celu"):
            alpha = float(a[1]) if len(a) > 1 else float(kw.get("alpha", 1.0))
            return formula(f"relu(a) + min({lit(alpha)} * (exp(a / {lit(alpha)}) - 1), 0)", [tensor_of(a[0], name)], name)
        if t.startswith("aten.log_sigmoid_forward"):
            return ("multi", [formula("-softplus(-a)", [tensor_of(a[0], name)], name), integer_result("log_sigmoid buffer", name)])
        if t.startswith("aten.log_sigmoid"):
            return formula("-softplus(-a)", [tensor_of(a[0], name)], name)
        # -- reductions
        if t.startswith(("aten.sum.dim_IntList", "aten.sum.default", "aten.sum.dim")):
            dims = a[1] if len(a) > 1 else kw.get("dim")
            keep = bool(a[2]) if len(a) > 2 else bool(kw.get("keepdim", False))
            axes = reduce_axes(a[0], dims, name, t)
            return reduce_chain("ReduceSum", tensor_of(a[0], name), axes, keep, name)
        if t.startswith(("aten.amax", "aten.amin")):
            dims = a[1] if len(a) > 1 else kw.get("dim", [])
            keep = bool(a[2]) if len(a) > 2 else bool(kw.get("keepdim", False))
            axes = reduce_axes(a[0], dims, name, t)
            return reduce_chain("ReduceMax" if "amax" in t else "ReduceMin", tensor_of(a[0], name), axes, keep, name)
        if t.startswith(("aten.max.default", "aten.min.default", "aten.prod.default")):
            raise UnsupportedOperator(t, name, "a reduction of every element, batch included, would mix examples")
        if t.startswith("aten.prod.dim_int"):
            keep = bool(a[2]) if len(a) > 2 else bool(kw.get("keepdim", False))
            axes = reduce_axes(a[0], a[1], name, t)
            return reduce_chain("ReduceProd", tensor_of(a[0], name), axes, keep, name)
        if t.startswith(("aten.var.", "aten.std.", "aten.var_mean.", "aten.std_mean.")):
            dims = a[1] if len(a) > 1 else kw.get("dim")
            keep = bool(kw.get("keepdim", a[3] if len(a) > 3 and isinstance(a[3], bool) else False))
            correction = kw.get("correction")
            if correction is None:
                unbiased = kw.get("unbiased", a[2] if len(a) > 2 and isinstance(a[2], bool) else True)
                correction = 1 if unbiased else 0
            correction = float(correction)
            x = tensor_of(a[0], name)
            axes = reduce_axes(a[0], dims, name, t)
            count = int(np.prod([shape_of(x)[d] for d in axes]))
            if count - correction <= 0:
                raise UnsupportedOperator(t, name, f"{count} elements for a correction of {correction}")
            mu = reduce_chain("ReduceMean", x, axes, True, f"{name}_mu")
            dev = b.emit("Sub", [x, mu], f"{name}_dev", shape=shape_of(x))
            sq = b.emit("Mul", [dev, dev], f"{name}_sq", shape=shape_of(x))
            var = reduce_chain("ReduceMean", sq, axes, keep, f"{name}_m2")
            factor = count / (count - correction)
            std = "std" in t.split(".")[1]
            expr = "sqrt(a * {k})" if std else "a * {k}"
            if factor == 1.0 and not std:
                out = var
            else:
                out = formula(expr.format(k=lit(factor)), [var], name)
            if t.startswith(("aten.var_mean", "aten.std_mean")):
                mean = mu if keep else reduce_chain("ReduceMean", x, axes, False, f"{name}_mean")
                return ("multi", [out, mean])
            return out
        if t.startswith("aten.cumsum"):
            if kw.get("dtype") is not None:
                raise UnsupportedOperator(t, name, "cumulative sum with a dtype change")
            d = axis_arg(a[0], a[1], name, t)
            x = tensor_of(a[0], name)
            return b.emit("CumSum", [x], name, {"axis": d}, shape_of(x))
        if t.startswith("aten.logsumexp"):
            keep = bool(a[2]) if len(a) > 2 else bool(kw.get("keepdim", False))
            axes = reduce_axes(a[0], a[1], name, t)
            if len(axes) != 1:
                raise UnsupportedOperator(t, name, "logsumexp over several axes")
            m, _, z = softmax_parts(tensor_of(a[0], name), axes[0], name)
            out = formula("log(a) + b", [z, m], f"{name}_k" if not keep else name)
            if keep:
                return out
            sh = list(shape_of(out))
            sh.pop(axes[0])
            return reshape(out, sh, name)
        if t.startswith(("aten._log_softmax", "aten.log_softmax")):
            x = tensor_of(a[0], name)
            d = axis_arg(a[0], a[1], name, t)
            _, shifted, z = softmax_parts(x, d, name)
            return formula("a - log(b)", [shifted, z], name)
        if t.startswith("aten.linalg_vector_norm"):
            ord_ = float(a[1]) if len(a) > 1 else float(kw.get("ord", 2))
            dims = a[2] if len(a) > 2 else kw.get("dim")
            keep = bool(a[3]) if len(a) > 3 else bool(kw.get("keepdim", False))
            x = tensor_of(a[0], name)
            axes = reduce_axes(a[0], dims, name, t)
            if ord_ == 2:
                sq = b.emit("Mul", [x, x], f"{name}_sq", shape=shape_of(x))
                return formula("sqrt(a)", [reduce_chain("ReduceSum", sq, axes, keep, f"{name}_ss")], name)
            if ord_ == 1:
                ab = formula("abs(a)", [x], f"{name}_abs")
                return reduce_chain("ReduceSum", ab, axes, keep, name)
            if ord_ == float("inf"):
                ab = formula("abs(a)", [x], f"{name}_abs")
                return reduce_chain("ReduceMax", ab, axes, keep, name)
            raise UnsupportedOperator(t, name, f"norm of order {ord_}")
        # -- padding, resizing, reordering
        if t.startswith("aten.constant_pad_nd"):
            x = tensor_of(a[0], name)
            value = float(a[2]) if len(a) > 2 else float(kw.get("value", 0.0))
            return pad_node(x, torch_pad_pairs(len(shape_of(x)), a[1]), "constant", value, name, a[0])
        if t.startswith(("aten.reflection_pad1d", "aten.reflection_pad2d", "aten.replication_pad1d",
                         "aten.replication_pad2d")):
            x = tensor_of(a[0], name)
            mode = "reflect" if "reflection" in t else "edge"
            return pad_node(x, torch_pad_pairs(len(shape_of(x)), a[1]), mode, None, name, a[0])
        if t.startswith(("aten.pad.default", "aten._pad_circular")):
            x = tensor_of(a[0], name)
            pairs = torch_pad_pairs(len(shape_of(x)), a[1])
            mode = "circular" if "circular" in t else (a[2] if len(a) > 2 else kw.get("mode", "constant"))
            if mode == "circular":
                if pairs[0] != (0, 0) and has_batch(a[0]):
                    raise UnsupportedOperator(t, name, "padding along the batch axis")
                return circular_pad(x, pairs, name)
            mode = {"constant": "constant", "reflect": "reflect", "replicate": "edge"}.get(mode)
            if mode is None:
                raise UnsupportedOperator(t, name, f"padding mode {a[2]!r}")
            value = a[3] if len(a) > 3 else kw.get("value")
            return pad_node(x, pairs, mode, value or 0.0, name, a[0])
        if t.startswith(("aten.upsample_nearest1d", "aten.upsample_nearest2d")):
            return resize(node, t, a, kw, "nearest")
        if t.startswith(("aten.upsample_bilinear2d", "aten.upsample_linear1d")):
            return resize(node, t, a, kw, "linear")
        if t.startswith(("aten._upsample_nearest_exact", "aten.upsample_bicubic", "aten._upsample_bilinear2d_aa")):
            raise UnsupportedOperator(t, name, "only nearest and (bi)linear resizing are translated")
        if t.startswith("aten.flip"):
            x = tensor_of(a[0], name)
            axes = [axis_arg(a[0], d, name, t) for d in a[1]]
            if not axes:
                return x
            return b.emit("Flip", [x], name, {"axes": sorted(set(axes))}, shape_of(x))
        if t.startswith("aten.roll"):
            x = tensor_of(a[0], name)
            shifts = [a[1]] if isinstance(a[1], int) else list(a[1])
            dims = a[2] if len(a) > 2 else kw.get("dims", [])
            dims = [dims] if isinstance(dims, int) else list(dims)
            if len(dims) != len(shifts):
                raise UnsupportedOperator(t, name, "roll of the flattened tensor")
            for k, (sft, d) in enumerate(zip(shifts, dims)):
                d = axis_arg(a[0], d, name, t)
                n = shape_of(x)[d]
                sft = int(sft) % n
                if sft:
                    tail = slice_(x, d, n - sft, n, f"{name}_t{k}")
                    head = slice_(x, d, 0, n - sft, f"{name}_h{k}")
                    x = concat_tree([tail, head], d, f"{name}_{k}" if k < len(dims) - 1 else name)
            return x
        if t.startswith("aten.repeat.default"):
            x = tensor_of(a[0], name)
            sx = shape_of(x)
            reps = [int(r) for r in a[1]]
            if len(reps) != len(sx):
                raise UnsupportedOperator(t, name, f"repeat {reps} of a tensor of rank {len(sx)}: new leading axes")
            if reps[0] != 1 and has_batch(a[0]):
                raise UnsupportedOperator(t, name, "repetition along the batch axis")
            for d, r in enumerate(reps):
                if r > 1:
                    x = concat_copies(x, d, r, f"{name}_{d}")
            return x
        # -- indexing by constant indices
        if t.startswith("aten.index_select"):
            d = axis_arg(a[0], a[1], name, t)
            idx = const_index(a[2], name, t)
            if idx.dim() != 1:
                raise UnsupportedOperator(t, name, "index of rank other than 1")
            return index_select(tensor_of(a[0], name), d, idx.tolist(), name)
        if t.startswith("aten.index.Tensor"):
            x = tensor_of(a[0], name)
            sx = shape_of(x)
            given = [(k, i) for k, i in enumerate(a[1]) if i is not None]
            if len(given) != 1:
                raise UnsupportedOperator(t, name, "indexing by several tensors at once")
            d, arg = given[0]
            if d == 0 and has_batch(a[0]):
                raise UnsupportedOperator(t, name, "indexing along the batch axis")
            idx = const_index(arg, name, t)
            flat = [int(i) for i in idx.flatten().tolist()]
            if idx.dim() <= 1:
                return index_select(x, d, flat, name)
            sel = index_select(x, d, flat, f"{name}_flat")
            target = list(sx[:d]) + [int(s) for s in idx.shape] + list(sx[d + 1:])
            if len(target) > 4:
                raise UnsupportedOperator(t, name, f"result of rank {len(target)}")
            return reshape(sel, target, name)
        if t.startswith("aten.embedding") and a[1] in const_nodes:
            # A constant index that is not a contiguous range (T5's relative position buckets):
            # IndexSelect of the rows, then the shape of the index.
            idx = const_index(a[1], name, t)
            flat = [int(i) for i in idx.flatten().tolist()]
            n = len(flat)
            contiguous = n > 0 and flat == list(range(flat[0], flat[0] + n))
            if contiguous or (n > 1 and len(set(flat)) == 1):
                return None  # original translation (Slice or broadcast row)
            w = tensor_of(a[0], name)
            sel = index_select(w, 0, flat, f"{name}_rows")
            target = [int(s) for s in idx.shape] + [shape_of(w)[-1]]
            if len(target) > 4:
                raise UnsupportedOperator(t, name, f"result of rank {len(target)}")
            return reshape(sel, target, name)
        if t.startswith("aten.gather"):
            d, line, _ = gather_as_select(a[0], a[1], a[2], name, t)
            x = tensor_of(a[0], name)
            if d == 0 and has_batch(a[0]):
                raise UnsupportedOperator(t, name, "gather along the batch axis")
            idx_shape = [int(v) for v in values[a[2]].shape]
            out = index_select(x, d, line, f"{name}_sel")
            # An index smaller than the input on another axis reads a slice of it.
            for k, (have, want) in enumerate(zip(shape_of(out), idx_shape)):
                if have != want:
                    if k == 0 and has_batch(a[0]):
                        raise UnsupportedOperator(t, name, "an index that does not cover the batch")
                    out = slice_(out, k, 0, want, f"{name}_s{k}")
            return out
        if t.startswith(("aten.index_add", "aten.scatter_add")):
            x = tensor_of(a[0], name)
            if t.startswith("aten.index_add"):
                d = axis_arg(a[0], a[1], name, t)
                idx = const_index(a[2], name, t)
                if idx.dim() != 1:
                    raise UnsupportedOperator(t, name, "index of rank other than 1")
                line = [int(i) % shape_of(x)[d] for i in idx.tolist()]
                src = tensor_of(a[3], name)
                alpha = kw.get("alpha", a[4] if len(a) > 4 else 1)
                if float(alpha) != 1.0:
                    src = formula(f"a * {lit(alpha)}", [src], f"{name}_alpha")
            else:
                d, line, other = gather_as_select(a[0], a[1], a[2], name, t)
                if d == 0 and has_batch(a[0]):
                    raise UnsupportedOperator(t, name, "scatter along the batch axis")
                src = tensor_of(a[3], name)
                ss = list(shape_of(src))
                want = [int(s) for s in values[a[2]].shape]
                if ss != want:
                    raise UnsupportedOperator(t, name, "src larger than the index")
                if [s for k, s in enumerate(shape_of(x)) if k != d] != other:
                    raise UnsupportedOperator(t, name, "an index smaller than the input on another axis")
            if not line:
                return x
            return b.emit("IndexAdd", [x, src], name, {"axis": d, "index": line}, shape_of(x))
        if t.startswith("aten.where.") and a[0] in const_nodes and not isinstance(a[0], (int, float)):
            # A constant condition that keeps everything, or nothing: the attention mask of a
            # sequence without padding (T5's encoder). The result is one of the two branches.
            cond = values[a[0]]
            if torch.is_tensor(cond) and cond.dtype == torch.bool and cond.numel():
                keep = a[1] if bool(cond.all()) else (a[2] if not bool(cond.any()) else None)
                if keep is not None and not is_number(keep) and keep not in const_nodes:
                    return tensor_of(keep, name)
        if t.startswith(("aten.index_put", "aten.masked_scatter", "aten.nonzero", "aten.unique",
                         "aten.one_hot")):
            raise UnsupportedOperator(t, name, "an output whose shape depends on the data, or an assignment "
                                      "by index, is outside the profile (ADR-0029)")
        return None

    # ---- gg-ops/8 (ADR-0029): integer tensors computed in the graph ----------------------------

    INT_DTYPES = ("torch.int64", "torch.int32", "torch.int16", "torch.int8", "torch.uint8")

    def is_int(arg) -> bool:
        val = getattr(arg, "meta", {}).get("val") if hasattr(arg, "meta") else None
        return val is not None and hasattr(val, "dtype") and str(val.dtype) in INT_DTYPES

    SHAPE_OPS = ("aten.view", "aten.reshape", "aten._unsafe_view", "aten.unsqueeze", "aten.squeeze",
                 "aten.select", "aten.slice", "aten.clone", "aten.contiguous", "aten.permute",
                 "aten.transpose", "aten.flatten", "aten._to_copy", "aten.to.", "aten.alias")

    def int_bound(arg) -> Optional[int]:
        """Exclusive bound of an integer tensor computed by an index operator, followed back through
        shape operators; None when it is not known (it comes from the data)."""
        seen = 0
        while hasattr(arg, "target") and seen < 64:
            seen += 1
            t = _target_name(arg.target)
            if t in ("getitem", "operator.getitem", "_operator.getitem"):
                arg = arg.args[0]
                continue
            if t.startswith(("aten.topk", "aten.sort", "aten.argsort", "aten.argmax", "aten.argmin",
                             "aten.max.dim", "aten.min.dim")):
                shp = list(arg.args[0].meta["val"].shape)
                if t.startswith("aten.topk"):
                    dim = arg.args[2] if len(arg.args) > 2 else arg.kwargs.get("dim", -1)
                elif t.endswith(".stable"):
                    dim = arg.kwargs.get("dim", -1)
                else:
                    dim = arg.args[1] if len(arg.args) > 1 else arg.kwargs.get("dim", -1)
                if dim is None:
                    return int(np.prod(shp))
                return int(shp[int(dim) % len(shp)])
            if t.startswith(SHAPE_OPS) and arg.args and hasattr(arg.args[0], "meta"):
                arg = arg.args[0]
                continue
            return None
        return None

    def distinct_along(arg, axis: int) -> bool:
        """True if the integer tensor is the indices of a topk, sort or argsort along `axis`."""
        while hasattr(arg, "target"):
            t = _target_name(arg.target)
            if t in ("getitem", "operator.getitem", "_operator.getitem"):
                arg = arg.args[0]
                continue
            if t.startswith(("aten.topk", "aten.sort", "aten.argsort")):
                shp = list(arg.args[0].meta["val"].shape)
                if t.startswith("aten.topk"):
                    dim = arg.args[2] if len(arg.args) > 2 else arg.kwargs.get("dim", -1)
                elif t.endswith(".stable"):
                    dim = arg.kwargs.get("dim", -1)
                else:
                    dim = arg.args[1] if len(arg.args) > 1 else arg.kwargs.get("dim", -1)
                return int(dim) % len(shp) == axis
            return False
        return False

    def index_op(op_name: str, x: str, axis: int, attrs: Dict[str, Any], out_axis_len: int, ctx: str) -> str:
        shp = list(shape_of(x))
        shp[axis] = out_axis_len
        return b.emit(op_name, [x], ctx, dict({"axis": axis}, **attrs), tuple(shp))

    def gather_elements(x: str, idx: str, axis: int, ctx: str) -> str:
        return b.emit("GatherElements", [x, idx], ctx, {"axis": axis}, shape_of(idx))

    def reshape_lead(x: str, target, batched: bool, ctx: str) -> str:
        """Reshape keeping the leading axis free (-1) when it carries the batch, possibly merged
        with other axes ([batch·T, …] in a mixture of experts)."""
        target = [int(v) for v in target]
        if tuple(target) == tuple(shape_of(x)):
            return x
        dims = ([-1] + target[1:]) if batched else target
        return b.emit("Reshape", [x], ctx, {"shape": dims}, tuple(target))

    def drop_axis(x: str, axis: int, ctx: str, batched: bool = True) -> str:
        shp = list(shape_of(x))
        shp.pop(axis)
        return reshape_lead(x, shp, batched, ctx)

    INT_OPS = ("aten.argmax", "aten.argmin", "aten.topk", "aten.sort", "aten.argsort", "aten.max.dim",
               "aten.min.dim", "aten.clamp", "aten.expand", "aten.cat", "aten.stack", "aten.repeat_interleave",
               "aten.chunk", "aten.split", "aten.unbind", "aten.lift_fresh_copy", "aten._assert") + SHAPE_OPS

    def translate_ops8(node, t: str, op: str, a, kw):
        """Translations added with `gg-ops/8` (ADR-0029). Returns the gg name (or an environment
        value), or None if `t` is not handled here."""
        name = node.name
        if is_int(node) and not t.startswith(INT_OPS) and any(
                is_int(x) for x in node.all_input_nodes if x not in const_nodes):
            raise UnsupportedOperator(t, name, "integer arithmetic is outside the profile: integers are indices, "
                                      "read by gathers and scatters (ADR-0029)")
        if t.startswith("aten.expand") and is_int(a[0]):
            # An integer tensor copied along axes of size 1: copies end to end (Concat).
            x = tensor_of(a[0], name)
            sx = list(shape_of(x))
            sizes = [int(v) for v in a[1]]
            if len(sizes) != len(sx):
                raise UnsupportedOperator(t, name, "expanding integers to a higher rank")
            for d, (have, want) in enumerate(zip(sx, sizes)):
                if want in (-1, have):
                    continue
                if have != 1 or (d == 0 and has_batch(a[0])):
                    raise UnsupportedOperator(t, name, f"expanding integers from {sx} to {sizes}")
                x = concat_copies(x, d, want, f"{name}_{d}")
            return x
        if t.startswith(("aten.argmax", "aten.argmin")):
            dim = a[1] if len(a) > 1 else kw.get("dim")
            keep = bool(a[2]) if len(a) > 2 else bool(kw.get("keepdim", False))
            if dim is None:
                raise UnsupportedOperator(t, name, "an index over every element, batch included, would mix examples")
            d = axis_arg(a[0], dim, name, t)
            x = tensor_of(a[0], name)
            opn = "ArgMax" if "argmax" in t else "ArgMin"
            i = index_op(opn, x, d, {"keepdims": 1}, 1, name if keep else f"{name}_k")
            return i if keep else drop_axis(i, d, name, has_batch(a[0]))
        if t.startswith(("aten.max.dim", "aten.min.dim")):
            # Values and indices, like PyTorch: the gradient reaches the index it returns.
            keep = bool(a[2]) if len(a) > 2 else bool(kw.get("keepdim", False))
            d = axis_arg(a[0], a[1], name, t)
            x = tensor_of(a[0], name)
            i = index_op("ArgMax" if "max" in t else "ArgMin", x, d, {"keepdims": 1}, 1, f"{name}_i")
            v = gather_elements(x, i, d, f"{name}_v")
            if keep:
                return ("multi", [v, i])
            bt = has_batch(a[0])
            return ("multi", [drop_axis(v, d, f"{name}_values", bt), drop_axis(i, d, f"{name}_indices", bt)])
        if t.startswith("aten.topk"):
            k = int(a[1])
            dim = a[2] if len(a) > 2 else kw.get("dim", -1)
            largest = bool(a[3]) if len(a) > 3 else bool(kw.get("largest", True))
            if not (bool(a[4]) if len(a) > 4 else bool(kw.get("sorted", True))):
                raise UnsupportedOperator(t, name, "topk(sorted=False): PyTorch's order is unspecified")
            d = axis_arg(a[0], dim, name, t)
            x = tensor_of(a[0], name)
            i = index_op("ArgTopK", x, d, {"k": k, "largest": 1 if largest else 0}, k, f"{name}_i")
            return ("multi", [gather_elements(x, i, d, f"{name}_v"), i])
        if t.startswith(("aten.sort", "aten.argsort")):
            stable_form = t.endswith(".stable")
            if stable_form:
                dim, desc = kw.get("dim", -1), bool(kw.get("descending", False))
            else:
                dim = a[1] if len(a) > 1 else kw.get("dim", -1)
                desc = bool(a[2]) if len(a) > 2 else bool(kw.get("descending", False))
            d = axis_arg(a[0], dim, name, t)
            x = tensor_of(a[0], name)
            i = index_op("ArgSort", x, d, {"descending": 1 if desc else 0}, shape_of(x)[d],
                         name if "argsort" in t else f"{name}_i")
            if "argsort" in t:
                return i
            return ("multi", [gather_elements(x, i, d, f"{name}_v"), i])
        if t.startswith(("aten._to_copy", "aten.to.")) and is_int(a[0]):
            want = kw.get("dtype") if t.startswith("aten._to_copy") else next((v for v in a[1:] if isinstance(v, torch.dtype)), kw.get("dtype"))
            if want is None or str(want) in INT_DTYPES:
                return tensor_of(a[0], name)
            if str(want) != "torch.float32":
                raise UnsupportedOperator(t, name, f"conversion of integers to {want}")
            x = tensor_of(a[0], name)
            return b.emit("Cast", [x], name, {"to": "f32"}, shape_of(x))
        if t.startswith("aten.clamp") and is_int(a[0]):
            lo = a[1] if len(a) > 1 else kw.get("min")
            hi = a[2] if len(a) > 2 else kw.get("max")
            bound = int_bound(a[0])
            if bound is not None and (lo is None or lo <= 0) and (hi is None or hi >= bound - 1):
                # Already in [0, bound): the clamp changes nothing.
                return tensor_of(a[0], name)
            raise UnsupportedOperator(t, name, "clamping integers whose range is not known (integer arithmetic is outside the profile)")
        if t.startswith("aten.gather") and a[2] not in const_nodes:
            d = axis_arg(a[0], a[1], name, t)
            return gather_elements(tensor_of(a[0], name), tensor_of(a[2], name), d, name)
        if t.startswith(("aten.scatter.src", "aten.scatter.value", "aten.scatter_add", "aten.scatter_reduce")) \
                and a[2] not in const_nodes:
            x, idx = tensor_of(a[0], name), tensor_of(a[2], name)
            d = axis_arg(a[0], a[1], name, t)
            sx, si = shape_of(x), shape_of(idx)
            if any(si[k] != sx[k] for k in range(len(sx)) if k != d):
                raise UnsupportedOperator(t, name, "an index smaller than the destination outside the scatter axis")
            if t.startswith("aten.scatter.value"):
                src = formula(f"a * 0 + {lit(a[3])}", [b.emit("Cast", [idx], f"{name}_c", {"to": "f32"}, si)], f"{name}_src")
            else:
                src = tensor_of(a[3], name)
                if tuple(shape_of(src)) != tuple(si):
                    for k in range(len(si)):
                        if shape_of(src)[k] != si[k]:
                            src = slice_(src, k, 0, si[k], f"{name}_s{k}")
            if t.startswith(("aten.scatter.src", "aten.scatter.value")):
                if kw.get("reduce"):
                    raise UnsupportedOperator(t, name, f"scatter with reduce={kw['reduce']!r}")
                if not distinct_along(a[2], d):
                    raise UnsupportedOperator(
                        t, name, "a scatter (assignment) whose positions are not known to be distinct: only "
                        "the indices of topk, sort or argsort along the same axis are accepted; repeated "
                        "positions would depend on the order of the writes (ADR-0029)")
                return b.emit("ScatterElements", [x, idx, src], name, {"axis": d, "reduction": "none"}, sx)
            if t.startswith("aten.scatter_add"):
                return b.emit("ScatterElements", [x, idx, src], name, {"axis": d}, sx)
            reduce = a[4] if len(a) > 4 else kw.get("reduce")
            include_self = bool(kw.get("include_self", a[5] if len(a) > 5 else True))
            if reduce not in ("sum", "mean"):
                raise UnsupportedOperator(t, name, f"scatter_reduce {reduce!r}: only sum and mean are executed (ADR-0029)")
            total = b.emit("ScatterElements", [formula("a * 0", [x], f"{name}_z"), idx, src], f"{name}_sum", {"axis": d}, sx)
            if reduce == "sum" and include_self:
                return b.emit("Add", [x, total], name, shape=sx)
            count = b.emit("ScatterElements", [formula("a * 0", [x], f"{name}_z1"), idx,
                                               formula("a * 0 + 1", [src], f"{name}_one")],
                           f"{name}_count", {"axis": d}, sx)
            # The counts are integers: min(c, 1) marks the written positions, exactly.
            if reduce == "sum":
                return formula("a * (1 - min(c, 1)) + b", [x, total, count], name)
            if include_self:
                return formula("(a + b) / (1 + c)", [x, total, count], name)
            return formula("a * (1 - min(c, 1)) + b / max(c, 1)", [x, total, count], name)
        computed_index = (
            t.startswith("aten.index.Tensor") and any(i is not None and i not in const_nodes for i in a[1])
        ) or (t.startswith("aten.index_select") and a[2] not in const_nodes)
        if computed_index:
            # Rows of a table chosen by a computed index: the weights of the experts a router picks.
            if t.startswith("aten.index.Tensor"):
                given = [(k, i) for k, i in enumerate(a[1]) if i is not None]
                ok = len(given) == 1 and given[0][0] == 0
                idx_arg = given[0][1]
            else:
                ok = int(a[1]) == 0
                idx_arg = a[2]
            if not ok or has_batch(a[0]):
                raise UnsupportedOperator(t, name, "indexing by a computed tensor is translated for the rows "
                                          "(axis 0) of a table without batch axis only (ADR-0029)")
            w, idx = tensor_of(a[0], name), tensor_of(idx_arg, name)
            out = tuple(shape_of(idx)) + tuple(shape_of(w)[1:])
            return b.emit("Gather", [w, idx], name, shape=out)
        if t.startswith("aten.repeat_interleave.self_int"):
            # Each slice repeated in a row: a new axis, copies placed end to end, then merged.
            x = tensor_of(a[0], name)
            n = int(a[1])
            sx = list(shape_of(x))
            d = int(a[2]) % len(sx) if len(a) > 2 and a[2] is not None else None
            if d is None:
                raise UnsupportedOperator(t, name, "repeat_interleave of the flattened tensor")
            if len(sx) + 1 > 4:
                raise UnsupportedOperator(t, name, "rank-5 intermediate")
            bt = has_batch(a[0])
            up = reshape_lead(x, sx[:d + 1] + [1] + sx[d + 1:], bt, f"{name}_u")
            cat = concat_copies(up, d + 1, n, name)
            target = list(sx)
            target[d] = sx[d] * n
            return reshape_lead(cat, target, bt, f"{name}_r")
        return None

    def translate_extra(node, t: str, a, kw):
        """Translations added with `gg-ops/6`: splits, shapes, element-wise functions, 1D
        convolution, group normalization, unrolled LSTM. Returns the gg name (or an environment
        value), or None if `t` is not handled here."""
        name = node.name
        op = t.split(".")[1] if "." in t else t
        done = translate_ops8(node, t, op, a, kw)
        if done is not None:
            return done
        done = translate_ops7(node, t, op, a, kw)
        if done is not None:
            return done
        if t.startswith("gpugoblin.formula"):
            # The researcher's custom operator (gpugoblin.formula): one node, as written.
            from .formula import check

            ins = [tensor_of(i, name) for i in a[0]]
            check(str(a[1]), len(ins))
            return formula(str(a[1]), ins, name)
        if t.startswith(("aten.chunk", "aten.split.Tensor", "aten.split_with_sizes", "aten.split.sizes", "aten.unbind")):
            x = tensor_of(a[0], name)
            sx = shape_of(x)
            if op == "unbind":
                axis = dim_arg(a[1] if len(a) > 1 else kw.get("dim", 0), len(sx))
                pieces = [1] * sx[axis]
            else:
                axis = dim_arg(a[2] if len(a) > 2 else kw.get("dim", 0), len(sx))
                size = sx[axis]
                if op == "chunk":
                    step = -(-size // int(a[1]))
                    pieces = [min(step, size - i) for i in range(0, size, step)]
                elif isinstance(a[1], (list, tuple)):
                    pieces = [int(v) for v in a[1]]
                else:
                    step = int(a[1])
                    pieces = [min(step, size - i) for i in range(0, size, step)]
            if axis == 0:
                raise UnsupportedOperator(t, name, "split along the batch axis")
            outs, start = [], 0
            for k, n in enumerate(pieces):
                piece = slice_(x, axis, start, start + n, f"{name}_{k}")
                if op == "unbind":
                    piece = reshape(piece, sx[:axis] + sx[axis + 1:], f"{name}_{k}r")
                outs.append(piece)
                start += n
            return ("multi", outs)
        if t.startswith("aten.select.int"):
            src = env.get(a[0])
            if isinstance(src, tuple) and src[0] == "layers":
                # h_n[k] of an LSTM: the state of layer k, already in [batch, H] form.
                return src[1][int(a[2])]
            x = tensor_of(a[0], name)
            sx = shape_of(x)
            axis = dim_arg(a[1], len(sx))
            i = int(a[2]) % sx[axis]
            if axis == 0:
                raise UnsupportedOperator(t, name, "selection along the batch axis")
            piece = slice_(x, axis, i, i + 1, f"{name}_s")
            return reshape(piece, sx[:axis] + sx[axis + 1:], name)
        if t.startswith("aten.squeeze"):
            src = env.get(a[0])
            if isinstance(src, tuple) and src[0] == "layers" and len(src[1]) == 1:
                return src[1][0]
            x = tensor_of(a[0], name)
            sx = shape_of(x)
            dims = a[1] if len(a) > 1 else None
            if dims is None:
                dims = [i for i, d in enumerate(sx) if d == 1 and i > 0]
            dims = [dims] if isinstance(dims, int) else list(dims)
            dims = {dim_arg(d, len(sx)) for d in dims if sx[dim_arg(d, len(sx))] == 1}
            if 0 in dims and batch == 1:
                dims.discard(0)
            return reshape(x, [d for i, d in enumerate(sx) if i not in dims], name)
        if t.startswith("aten.stack"):
            parts = [tensor_of(p, name) for p in a[0]]
            s0 = shape_of(parts[0])
            axis = dim_arg(a[1] if len(a) > 1 else kw.get("dim", 0), len(s0) + 1)
            if axis == 0:
                raise UnsupportedOperator(t, name, "stacking along the batch axis")
            if len(s0) + 1 > 4:
                raise UnsupportedOperator(t, name, "rank-5 stacking")
            target = list(s0[:axis]) + [1] + list(s0[axis:])
            ups = [reshape(p, target, f"{name}_u{k}") for k, p in enumerate(parts)]
            return concat_tree(ups, axis, name)
        if t.startswith(("aten.type_as", "aten.lift_fresh_copy")):
            return tensor_of(a[0], name)
        if op in ELEMENTWISE_FORMULAS and not t.startswith("aten.__"):
            return formula(ELEMENTWISE_FORMULAS[op], [tensor_of(a[0], name)], name)
        if t.startswith("aten.softplus"):
            beta = float(a[1]) if len(a) > 1 else float(kw.get("beta", 1.0))
            x = tensor_of(a[0], name)
            # Above the threshold (20), PyTorch returns x: the gap with the stable form is < 3e-9.
            expr = "softplus(a)" if beta == 1.0 else f"softplus(a * {beta!r}) / {beta!r}"
            return formula(expr, [x], name)
        if t.startswith(("aten.clamp", "aten.clip", "aten.hardtanh")):
            x = tensor_of(a[0], name)
            lo = a[1] if len(a) > 1 else kw.get("min", -1.0 if "hardtanh" in t else None)
            hi = a[2] if len(a) > 2 else kw.get("max", 1.0 if "hardtanh" in t else None)
            if op == "clamp_min":
                lo, hi = a[1], None
            elif op == "clamp_max":
                lo, hi = None, a[1]
            if any(v is not None and not isinstance(v, (int, float)) for v in (lo, hi)):
                raise UnsupportedOperator(t, name, "tensor bounds")
            expr = "a"
            if lo is not None:
                expr = f"max({expr}, {float(lo)!r})"
            if hi is not None:
                expr = f"min({expr}, {float(hi)!r})"
            return formula(expr, [x], name)
        if t.startswith("aten.leaky_relu"):
            slope = float(a[1]) if len(a) > 1 else float(kw.get("negative_slope", 0.01))
            return formula(f"relu(a) - relu(-a) * {slope!r}", [tensor_of(a[0], name)], name)
        if t.startswith("aten.elu"):
            # elu(x, alpha, scale, input_scale): SELU and CELU arrive in this form.
            alpha = float(a[1]) if len(a) > 1 else float(kw.get("alpha", 1.0))
            scale = float(a[2]) if len(a) > 2 else float(kw.get("scale", 1.0))
            in_scale = float(a[3]) if len(a) > 3 else float(kw.get("input_scale", 1.0))
            neg = "min(exp(a) - 1, 0)" if in_scale == 1.0 else f"min(exp(a * {in_scale!r}) - 1, 0)"
            expr = f"relu(a) + {neg} * {alpha!r}"
            if scale != 1.0:
                expr = f"({expr}) * {scale!r}"
            return formula(expr, [tensor_of(a[0], name)], name)
        if t.startswith(("aten.maximum", "aten.minimum")):
            fn = "max" if "maximum" in t else "min"
            return formula(f"{fn}(a, b)", [tensor_of(a[0], name), tensor_of(a[1], name)], name)
        if t.startswith("aten.pow.Tensor_Scalar"):
            e = float(a[1])
            x = tensor_of(a[0], name)
            if e == 1.0:
                return x
            if e == 2.0:
                return b.emit("Mul", [x, x], name, shape=shape_of(x))
            return formula(f"a^{e!r}", [x], name)
        if t.startswith("aten.pow.Scalar"):
            base = float(a[0])
            if not base > 0:
                raise UnsupportedOperator(t, name, f"base {base}: only a positive base is translated")
            return formula(f"exp(a * {float(np.log(base))!r})", [tensor_of(a[1], name)], name)
        if t.startswith("aten.expand"):
            src = env.get(a[0])
            if isinstance(src, tuple) and src[0] == "unsq":
                return None  # head repetition, handled further down
            x = tensor_of(a[0], name)
            sx = shape_of(x)
            sizes = [int(v) for v in a[1]]
            pad = len(sizes) - len(sx)
            full = (1,) * pad + tuple(sx)
            sizes = [full[i] if d == -1 else d for i, d in enumerate(sizes)]
            if tuple(sizes) == tuple(sx):
                return x
            if any(f not in (1, s) for f, s in zip(full, sizes)):
                raise UnsupportedOperator(t, name, f"expand from {sx} to {sizes}")
            ins, terms = [x], "a"
            rest = [1] + sizes[1:]
            if tuple(rest) != (1,) + tuple(full[1:]):
                ins.append(zeros_param(rest, name))
                terms += " + b"
            if sizes[0] != full[0]:
                if sizes[0] != batch:
                    raise UnsupportedOperator(t, name, "expand of a leading axis that is not the batch")
                ins.append(batch_ref(len(sizes), name))
                terms += f" + {'bc'[len(ins) - 2]} * 0"
            return formula(terms, ins, name)
        if t.startswith("aten.mean.dim"):
            x = tensor_of(a[0], name)
            sx = shape_of(x)
            dims = a[1] if len(a) > 1 else kw.get("dim")
            dims = [dims] if isinstance(dims, int) else list(dims or [])
            keep = bool(a[2]) if len(a) > 2 else bool(kw.get("keepdim", False))
            if len(dims) < 2 or kw.get("dtype") is not None:
                return None  # a single axis: original translation
            out, cur = list(sx), x
            for d in sorted({dim_arg(d, len(sx)) for d in dims}, reverse=True):
                if keep:
                    out[d] = 1
                else:
                    out.pop(d)
                cur = b.emit("ReduceMean", [cur], f"{name}_{d}", {"axis": d, "keepdims": 1 if keep else 0}, tuple(out))
            return cur
        if t.startswith(("aten.adaptive_avg_pool2d", "aten._adaptive_avg_pool2d")):
            x = tensor_of(a[0], name)
            sx = shape_of(x)
            oh, ow = as_pair(a[1])
            h, w = sx[-2], sx[-1]
            if (oh, ow) == (1, 1):
                r = b.emit("ReduceMean", [x], f"{name}_w", {"axis": 3, "keepdims": 1}, tuple(sx[:3]) + (1,))
                return b.emit("ReduceMean", [r], name, {"axis": 2, "keepdims": 1}, tuple(sx[:2]) + (1, 1))
            if h % oh or w % ow:
                raise UnsupportedOperator(t, name, f"output {oh}×{ow} does not divide {h}×{w}")
            k = [h // oh, w // ow]
            return b.emit("AvgPool2d", [x], name, {"kernel": k, "stride": k, "padding": [0, 0]}, tuple(sx[:2]) + (oh, ow))
        if t.startswith(("aten.conv1d", "aten.convolution")) and len(shape_of(tensor_of(a[1], name))) == 3:
            return conv1d(node, t, a)
        if t.startswith(("aten.group_norm", "aten.native_group_norm")):
            return group_norm(node, t, a, kw)
        if t.startswith("aten.lstm.input"):
            return lstm(node, t, a)
        return None

    def conv1d(node, t: str, a):
        """1D convolution: `[N, C, L]` seen as `[N, C, 1, L]`, kernel `[O, I, 1, K]`, then the
        profile's `Conv2d`. Same products, same summation order."""
        name = node.name
        x, w = tensor_of(a[0], name), tensor_of(a[1], name)
        sx, sw = shape_of(x), shape_of(w)
        bias = a[2] if len(a) > 2 else None
        if t.startswith("aten.convolution"):
            stride, padding, dilation = int(a[3][0]), a[4][0], int(a[5][0])
            if bool(a[6]):
                raise UnsupportedOperator(t, name, "transposed convolution")
            groups = int(a[8])
        else:
            stride = int(a[3][0]) if len(a) > 3 else 1
            padding = a[4] if len(a) > 4 else 0
            dilation = int(a[5][0]) if len(a) > 5 else 1
            groups = int(a[6]) if len(a) > 6 else 1
        k = sw[2]
        if isinstance(padding, str):
            if padding == "valid":
                padding = 0
            elif padding == "same" and stride == 1 and (dilation * (k - 1)) % 2 == 0:
                padding = dilation * (k - 1) // 2
            else:
                raise UnsupportedOperator(t, name, f"asymmetric padding '{padding}'")
        elif isinstance(padding, (list, tuple)):
            padding = int(padding[0])
        padding = int(padding)
        x4 = reshape(x, (sx[0], sx[1], 1, sx[2]), f"{name}_x4")
        w4 = reshape(w, (sw[0], sw[1], 1, k), f"{name}_w4")
        ins = [x4, w4]
        if bias is not None:
            ins.append(tensor_of(bias, name))
        length = (sx[2] + 2 * padding - dilation * (k - 1) - 1) // stride + 1
        attrs = {"stride": [1, stride], "padding": [0, padding], "dilation": [1, dilation], "groups": groups}
        y = b.emit("Conv2d", ins, f"{name}_2d", attrs, (sx[0], sw[0], 1, length))
        return reshape(y, (sx[0], sw[0], length), name)

    def group_norm(node, t: str, a, kw):
        """Group normalization, written with the profile: mean and variance per group (`ReduceMean`),
        `Formula` for `rsqrt(var + ε)`, then per-channel gain and bias."""
        name = node.name
        x = tensor_of(a[0], name)
        sx = shape_of(x)
        if t.startswith("aten.native_group_norm"):
            # (input, weight, bias, N, C, HxW, group, eps)
            weight, bias, groups, eps = a[1], a[2], int(a[6]), float(a[7])
        else:
            groups = int(a[1])
            weight = a[2] if len(a) > 2 else kw.get("weight")
            bias = a[3] if len(a) > 3 else kw.get("bias")
            eps = float(a[4]) if len(a) > 4 else float(kw.get("eps", 1e-5))
        c = sx[1]
        per = int(np.prod(sx[1:])) // groups
        g = reshape(x, (sx[0], groups, per), f"{name}_g")
        mu = b.emit("ReduceMean", [g], f"{name}_mu", {"axis": 2, "keepdims": 1}, (sx[0], groups, 1))
        d = b.emit("Sub", [g, mu], f"{name}_d", shape=shape_of(g))
        d2 = b.emit("Mul", [d, d], f"{name}_d2", shape=shape_of(g))
        var = b.emit("ReduceMean", [d2], f"{name}_var", {"axis": 2, "keepdims": 1}, (sx[0], groups, 1))
        r = formula(f"rsqrt(a + {eps!r})", [var], f"{name}_r")
        n = b.emit("Mul", [d, r], f"{name}_n", shape=shape_of(g))
        out = reshape(n, sx, f"{name}_x")
        tail = (1,) * (len(sx) - 2)
        if weight is not None:
            gamma = reshape(tensor_of(weight, name), (c,) + tail, f"{name}_gamma")
            out = b.emit("Mul", [out, gamma], f"{name}_scaled", shape=sx)
        if bias is not None:
            beta = reshape(tensor_of(bias, name), (c,) + tail, f"{name}_beta")
            out = b.emit("Add", [out, beta], name if not t.startswith("aten.native") else f"{name}_out", shape=sx)
        return ("multi", out) if t.startswith("aten.native") else out

    def lstm(node, t: str, a):
        """LSTM unrolled over time: the sequence has a fixed length in the description, and each step
        is written with the profile (`MatMul`, `Slice`, `Formula`). Stacked layers, batch first
        (`batch_first=True`), unidirectional, no dropout between layers."""
        name = node.name
        # (input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first)
        x = tensor_of(a[0], name)
        hx, params_in = list(a[1]), list(a[2])
        has_bias, layers = bool(a[3]), int(a[4])
        dropout, train, bidir, batch_first = float(a[5]), bool(a[6]), bool(a[7]), bool(a[8])
        if bidir:
            raise UnsupportedOperator(t, name, "bidirectional LSTM")
        if not batch_first:
            raise UnsupportedOperator(t, name, "sequence first (batch_first=False): the batch must be the first axis")
        if dropout and train:
            raise UnsupportedOperator(t, name, "dropout between layers")
        per = 4 if has_bias else 2
        if len(params_in) != per * layers:
            raise UnsupportedOperator(t, name, "projection (proj_size) or unexpected parameters")
        sx = shape_of(x)
        steps = sx[1]
        inits = []
        for k, h0 in enumerate(hx):
            u = uniform_value(h0) if h0 in const_nodes else None
            if u == 0.0:
                inits.append(None)
            elif h0 in const_nodes:
                raise UnsupportedOperator(t, name, "nonzero constant initial state")
            else:
                raise UnsupportedOperator(t, name, "initial state provided: only the zero state is translated")
        seq, hs, cs = x, [], []
        for layer in range(layers):
            w_ih, w_hh = (tensor_of(params_in[per * layer + i], name) for i in range(2))
            four_h = shape_of(w_ih)[0]
            hid = four_h // 4
            pre = f"{name}.l{layer}"
            w_ihT = b.emit("Transpose", [w_ih], f"{pre}.wihT", {"perm": [1, 0]}, shape_of(w_ih)[::-1])
            w_hhT = b.emit("Transpose", [w_hh], f"{pre}.whhT", {"perm": [1, 0]}, shape_of(w_hh)[::-1])
            s_seq = shape_of(seq)
            gx = b.emit("MatMul", [seq, w_ihT], f"{pre}.gx", shape=(s_seq[0], steps, four_h))
            if has_bias:
                b_ih, b_hh = (tensor_of(params_in[per * layer + i], name) for i in (2, 3))
                gx = b.emit("Add", [gx, b_ih], f"{pre}.gxb", shape=shape_of(gx))
                gx = b.emit("Add", [gx, b_hh], f"{pre}.gxbb", shape=shape_of(gx))
            h = c = None
            outs = []
            for step in range(steps):
                p = f"{pre}.t{step}"
                g = reshape(slice_(gx, 1, step, step + 1, f"{p}.gs"), (s_seq[0], four_h), f"{p}.g")
                if h is not None:
                    gh = b.emit("MatMul", [h, w_hhT], f"{p}.gh", shape=(s_seq[0], four_h))
                    g = b.emit("Add", [g, gh], f"{p}.gg", shape=(s_seq[0], four_h))
                i_, f_, g_, o_ = (slice_(g, 1, k * hid, (k + 1) * hid, f"{p}.{n}") for k, n in enumerate("ifgo"))
                if c is None:
                    c = formula("sigmoid(a) * tanh(b)", [i_, g_], f"{p}.c")
                else:
                    c = formula("sigmoid(a) * b + sigmoid(c) * tanh(d)", [f_, c, i_, g_], f"{p}.c")
                h = formula("sigmoid(a) * tanh(b)", [o_, c], f"{p}.h")
                outs.append(reshape(h, (s_seq[0], 1, hid), f"{p}.h3"))
            seq = concat_tree(outs, 1, f"{pre}.out")
            hs.append(h)
            cs.append(c)
        return ("multi", [seq, ("layers", hs), ("layers", cs)])

    outputs: List[str] = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            continue
        if node.op == "output":
            outs = node.args[0]
            outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
            outputs = [tensor_of(o, "output") for o in outs]
            continue
        if node.op == "get_attr":
            env[node] = ("constant", node.target)
            continue
        if node in const_nodes:
            # Computed once here, never emitted: consumed by the causal mask or turned into a scalar.
            continue
        if node.op != "call_function":
            raise UnsupportedOperator(str(node.op), node.name)
        t = _target_name(node.target)
        a = node.args
        kw = node.kwargs
        try:
            # ---- tuple access for operators with several outputs
            if t in ("getitem", "operator.getitem", "_operator.getitem"):
                src = env[a[0]]
                if isinstance(src, tuple) and src[0] == "multi":
                    k = int(a[1])
                    if isinstance(src[1], list):
                        env[node] = src[1][k]
                    else:
                        env[node] = src[1] if k == 0 else ("dropped", t)
                    continue
                env[node] = ("dropped", t)
                continue
            # In-place variant (`add_`, `relu_`): same computation, the captured graph has no aliasing.
            parts = t.split(".")
            if len(parts) > 1 and parts[1].endswith("_") and not parts[1].startswith("_"):
                parts[1] = parts[1][:-1]
                t = ".".join(parts)
            done = translate_extra(node, t, a, kw)
            if done is not None:
                env[node] = done
                continue
            if t.startswith(("aten._assert_tensor_metadata", "aten._assert_scalar", "aten.sym_constrain_range", "aten._assert_async")):
                # torch.export checks, no computation.
                env[node] = ("dropped", t)
                continue
            if t.startswith("aten.unsqueeze"):
                x = tensor_of(a[0], node.name)
                sx = shape_of(x)
                dim = int(a[1]) % (len(sx) + 1)
                if len(sx) + 1 > 4:
                    # Rank 5: only as a step of a head repetition (expand then reshape).
                    env[node] = ("unsq", x, dim)
                    continue
                target = list(sx[:dim]) + [1] + list(sx[dim:])
                env[node] = b.emit("Reshape", [x], node.name, {"shape": batch_reshape(target, sx)}, tuple(target))
                continue
            if t.startswith("aten.expand"):
                src = env.get(a[0])
                sizes = [int(v) for v in a[1]]
                if isinstance(src, tuple) and src[0] == "unsq":
                    x, dim = src[1], src[2]
                    sx = shape_of(x)
                    full = list(sx[:dim]) + [1] + list(sx[dim:])
                    sizes = [full[i] if d == -1 else d for i, d in enumerate(sizes)]
                    if [d for i, d in enumerate(sizes) if i != dim] != [d for i, d in enumerate(full) if i != dim]:
                        raise UnsupportedOperator(t, node.name, "expand along an axis other than the inserted one")
                    env[node] = ("rep", x, dim, sizes[dim])
                    continue
                x = tensor_of(a[0], node.name)
                sx = shape_of(x)
                sizes = [sx[i] if d == -1 else d for i, d in enumerate(sizes)]
                if tuple(sizes) != tuple(sx):
                    raise UnsupportedOperator(t, node.name, "expand that changes the shape; only head repetition (GQA) is translated")
                env[node] = x
                continue
            if t.startswith(("aten.view", "aten.reshape", "aten._unsafe_view")) and isinstance(env.get(a[0]), tuple) and env[a[0]][0] == "rep":
                _, x, dim, n = env[a[0]]
                sx = shape_of(x)
                if dim == 0:
                    raise UnsupportedOperator(t, node.name, "repetition along the batch axis")
                want = list(sx)
                want[dim - 1] = sx[dim - 1] * n
                concrete = [batch if d == -1 else int(d) for d in a[1]]
                if concrete != want:
                    raise UnsupportedOperator(t, node.name, f"fold {list(a[1])} of a repetition of {sx}: only merging the inserted axis with the previous one is translated")
                env[node] = repeat_heads(x, dim - 1, n, node.name)
                continue
            if t.startswith("aten.silu"):
                x = tensor_of(a[0], node.name)
                sig = b.emit("Sigmoid", [x], f"{node.name}_sig", shape=shape_of(x))
                env[node] = b.emit("Mul", [x, sig], node.name, shape=shape_of(x))
                continue
            if t.startswith("aten.pow.Tensor_Scalar"):
                x = tensor_of(a[0], node.name)
                e = float(a[1])
                if e == 2.0:
                    env[node] = b.emit("Mul", [x, x], node.name, shape=shape_of(x))
                elif e == 1.0:
                    env[node] = x
                else:
                    raise UnsupportedOperator(t, node.name, f"power {e}: only the square and the identity are translated")
                continue
            if t.startswith("aten.rsqrt"):
                x = tensor_of(a[0], node.name)
                env[node] = b.emit("Rsqrt", [x], node.name, shape=shape_of(x))
                continue
            if t.startswith("aten.mean.dim"):
                x = tensor_of(a[0], node.name)
                sx = shape_of(x)
                dims = a[1] if len(a) > 1 else kw.get("dim")
                dims = [dims] if isinstance(dims, int) else list(dims or [])
                keep = bool(a[2]) if len(a) > 2 else bool(kw.get("keepdim", False))
                if len(dims) != 1 or kw.get("dtype") is not None:
                    raise UnsupportedOperator(t, node.name, "mean over several axes")
                axis = int(dims[0]) % len(sx)
                out = list(sx)
                if keep:
                    out[axis] = 1
                else:
                    out.pop(axis)
                env[node] = b.emit("ReduceMean", [x], node.name, {"axis": axis, "keepdims": 1 if keep else 0}, tuple(out))
                continue
            if t.startswith("aten.linear"):
                x = tensor_of(a[0], node.name)
                w = tensor_of(a[1], node.name)
                bias = maybe(a[2]) if len(a) > 2 else None
                wt = b.emit("Transpose", [w], f"{node.name}_wT", {"perm": [1, 0]}, shape_of(w)[::-1])
                out = b.emit("MatMul", [x, wt], node.name, shape=shape_of(x)[:-1] + shape_of(w)[:1])
                if bias is not None:
                    out = b.emit("Add", [out, bias], f"{node.name}_b", shape=shape_of(out))
                env[node] = out
            elif t.startswith("aten.addmm"):
                bias, x, w = (tensor_of(a[i], node.name) for i in range(3))
                mm = b.emit("MatMul", [x, w], node.name, shape=shape_of(x)[:-1] + shape_of(w)[-1:])
                env[node] = b.emit("Add", [mm, bias], f"{node.name}_b", shape=shape_of(mm))
            elif t.startswith(("aten.mm", "aten.matmul", "aten.bmm")):
                x, w = tensor_of(a[0], node.name), tensor_of(a[1], node.name)
                sx, sw = shape_of(x), shape_of(w)
                if len(sx) < 2 or len(sw) < 2:
                    raise UnsupportedOperator(t, node.name, "product with a vector (rank 1)")
                # Broadcast batch axes, as in `torch.matmul`: [64, 2] × [N, 2, 8] -> [N, 64, 8].
                bx, bw = sx[:-2], sw[:-2]
                r = max(len(bx), len(bw))
                bx, bw = (1,) * (r - len(bx)) + tuple(bx), (1,) * (r - len(bw)) + tuple(bw)
                lead = tuple(max(p, q) for p, q in zip(bx, bw))
                env[node] = b.emit("MatMul", [x, w], node.name, shape=lead + (sx[-2], sw[-1]))
            elif t.startswith("aten.t."):
                x = tensor_of(a[0], node.name)
                env[node] = b.emit("Transpose", [x], node.name, {"perm": [1, 0]}, shape_of(x)[::-1])
            elif t.startswith("aten.transpose"):
                x = tensor_of(a[0], node.name)
                r = len(shape_of(x))
                i, j = int(a[1]) % r, int(a[2]) % r
                perm = list(range(r))
                perm[i], perm[j] = perm[j], perm[i]
                env[node] = b.emit("Transpose", [x], node.name, {"perm": perm}, tuple(shape_of(x)[p] for p in perm))
            elif t.startswith("aten.permute"):
                x = tensor_of(a[0], node.name)
                perm = [int(p) % len(shape_of(x)) for p in a[1]]
                env[node] = b.emit("Transpose", [x], node.name, {"perm": perm}, tuple(shape_of(x)[p] for p in perm))
            elif t.startswith(("aten.view", "aten.reshape", "aten._unsafe_view")):
                x = tensor_of(a[0], node.name)
                # A -1 can stand for any dimension, not only the batch: it is resolved from the
                # element count, and then only the batch becomes -1 again.
                original = [int(d) for d in a[1]]
                concrete = list(original)
                if -1 in concrete:
                    known = int(np.prod([d for d in concrete if d != -1]))
                    concrete[concrete.index(-1)] = int(np.prod(shape_of(x))) // max(known, 1)
                if tuple(concrete) == tuple(shape_of(x)):
                    # A view that changes nothing (`ids.view(-1, T)` on ids [batch, T]).
                    env[node] = x
                    continue
                # In the leading position, -1 absorbs the batch (batch·T, for example): it stays -1 for the executor.
                dims = original if original[0] == -1 else batch_reshape(concrete, shape_of(x))
                env[node] = b.emit("Reshape", [x], node.name, {"shape": dims}, tuple(concrete))
            elif t.startswith("aten.flatten"):
                x = tensor_of(a[0], node.name)
                start = int(a[1]) if len(a) > 1 else 1
                end = int(a[2]) if len(a) > 2 else -1
                if end not in (-1, len(shape_of(x)) - 1):
                    raise UnsupportedOperator(t, node.name, "partial flatten (end_dim) cannot be expressed")
                s = shape_of(x)
                start = start % len(s) if s else start
                if start > 1:
                    # `Flatten` returns rank 2; flatten(2) keeps the leading axes: it is a Reshape.
                    env[node] = reshape(x, tuple(s[:start]) + (int(np.prod(s[start:])),), node.name)
                    continue
                head = int(np.prod(s[:start])) if start else 1
                env[node] = b.emit("Flatten", [x], node.name, {"axis": start}, (head, int(np.prod(s[start:]))))
            elif t.startswith("aten.relu"):
                x = tensor_of(a[0], node.name)
                env[node] = b.emit("Relu", [x], node.name, shape=shape_of(x))
            elif t.startswith("aten.gelu"):
                x = tensor_of(a[0], node.name)
                attrs = {"approximate": kw["approximate"]} if kw.get("approximate", "none") != "none" else None
                env[node] = b.emit("Gelu", [x], node.name, attrs, shape_of(x))
            elif t.startswith("aten.sigmoid"):
                x = tensor_of(a[0], node.name)
                env[node] = b.emit("Sigmoid", [x], node.name, shape=shape_of(x))
            elif t.startswith("aten.tanh"):
                x = tensor_of(a[0], node.name)
                env[node] = b.emit("Tanh", [x], node.name, shape=shape_of(x))
            elif t.startswith(("aten._softmax", "aten.softmax")):
                x = tensor_of(a[0], node.name)
                env[node] = b.emit("Softmax", [x], node.name, {"axis": int(a[1])}, shape_of(x))
            elif t.startswith(("aten.layer_norm", "aten.native_layer_norm")):
                x = tensor_of(a[0], node.name)
                normalized = list(a[1])
                if len(normalized) != 1:
                    raise UnsupportedOperator(t, node.name, "normalization over several axes")
                ins = [x]
                if len(a) > 2 and a[2] is not None:
                    ins.append(tensor_of(a[2], node.name))
                if len(a) > 3 and a[3] is not None:
                    ins.append(tensor_of(a[3], node.name))
                eps = float(a[4]) if len(a) > 4 else float(kw.get("eps", 1e-5))
                out = b.emit("LayerNorm", ins, node.name, {"axis": -1, "epsilon": eps}, shape_of(x))
                env[node] = ("multi", out) if t.startswith("aten.native") else out
            elif t.startswith("aten.rms_norm"):
                x = tensor_of(a[0], node.name)
                ins = [x]
                if len(a) > 2 and a[2] is not None:
                    ins.append(tensor_of(a[2], node.name))
                eps = float(a[3]) if len(a) > 3 and a[3] is not None else 1e-5
                env[node] = b.emit("RMSNorm", ins, node.name, {"axis": -1, "epsilon": eps}, shape_of(x))
            elif t.startswith(("aten.conv2d", "aten.convolution")):
                x, w = tensor_of(a[0], node.name), tensor_of(a[1], node.name)
                ins = [x, w]
                if len(a) > 2 and a[2] is not None:
                    ins.append(tensor_of(a[2], node.name))
                stride = as_pair(a[3]) if len(a) > 3 else [1, 1]
                padding = as_pair(a[4]) if len(a) > 4 else [0, 0]
                dilation = as_pair(a[5]) if len(a) > 5 else [1, 1]
                if t.startswith("aten.convolution") and len(a) > 6 and bool(a[6]):
                    raise UnsupportedOperator(t, node.name, "transposed convolution")
                groups = int(a[8]) if t.startswith("aten.convolution") and len(a) > 8 else (int(a[6]) if len(a) > 6 else 1)
                attrs = {"stride": stride, "padding": padding, "dilation": dilation, "groups": groups}
                sx, sw = shape_of(x), shape_of(w)
                h = (sx[2] + 2 * padding[0] - dilation[0] * (sw[2] - 1) - 1) // stride[0] + 1
                wd = (sx[3] + 2 * padding[1] - dilation[1] * (sw[3] - 1) - 1) // stride[1] + 1
                env[node] = b.emit("Conv2d", ins, node.name, attrs, (sx[0], sw[0], h, wd))
            elif t.startswith(("aten.max_pool2d", "aten.avg_pool2d")):
                x = tensor_of(a[0], node.name)
                kernel = as_pair(a[1])
                stride = as_pair(a[2]) if len(a) > 2 and a[2] else kernel
                padding = as_pair(a[3]) if len(a) > 3 else [0, 0]
                op = "MaxPool2d" if "max" in t else "AvgPool2d"
                sx = shape_of(x)
                h = (sx[2] + 2 * padding[0] - kernel[0]) // stride[0] + 1
                wd = (sx[3] + 2 * padding[1] - kernel[1]) // stride[1] + 1
                out = b.emit(op, [x], node.name, {"kernel": kernel, "stride": stride, "padding": padding}, (sx[0], sx[1], h, wd))
                env[node] = ("multi", out) if "with_indices" in t else out
            elif t.startswith(("aten.batch_norm", "aten._native_batch_norm", "aten.native_batch_norm", "aten.cudnn_batch_norm")):
                x = tensor_of(a[0], node.name)
                gamma = maybe(a[1]) if len(a) > 1 else None
                beta = maybe(a[2]) if len(a) > 2 else None
                if gamma is None or beta is None:
                    raise UnsupportedOperator(t, node.name, "BatchNorm without gain or bias")
                eps = float(a[7]) if len(a) > 7 else float(kw.get("eps", 1e-5))
                out = b.emit("BatchNorm2d", [x, gamma, beta], node.name, {"epsilon": eps}, shape_of(x))
                env[node] = ("multi", out) if "native" in t or "cudnn" in t else out
            elif t.startswith("aten.cat"):
                parts = [tensor_of(p, node.name) for p in a[0]
                         if not (isinstance(env.get(p), tuple) and env[p][0] == "empty")]
                if len(parts) == 1:
                    env[node] = parts[0]
                    continue
                axis = int(a[1]) if len(a) > 1 else 0
                s = list(shape_of(parts[0]))
                # The profile expects a non-negative axis: dim=-1 becomes the last axis.
                axis = axis % len(s) if s else axis
                if s:
                    s[axis] = sum(shape_of(p)[axis] for p in parts)
                env[node] = b.emit("Concat", parts, node.name, {"axis": axis}, tuple(s))
            elif t.startswith("aten.embedding") and a[1] in const_nodes:
                # Learned positions: `wpe(arange(t))` selects contiguous rows of a trained table.
                # All rows → the table itself; some of them → Slice.
                w = tensor_of(a[0], node.name)
                idx = values[a[1]].flatten().to(dtype=__import__("torch").int64)
                n = int(idx.numel())
                start = int(idx[0]) if n else 0
                contiguous = n > 0 and bool((idx == __import__("torch").arange(start, start + n)).all())
                if n > 1 and bool((idx == idx[0]).all()):
                    # The same row everywhere (BERT's all-zero token types): that row, broadcast
                    # over the batch and the sequence.
                    row = int(idx[0])
                    width = shape_of(w)[-1]
                    one = slice_(w, 0, row, row + 1, f"{node.name}_row") if shape_of(w)[0] > 1 else w
                    env[node] = reshape(one, (1,) * values[a[1]].dim() + (width,), node.name)
                    continue
                if not contiguous or values[a[1]].dim() > 2 or (values[a[1]].dim() == 2 and values[a[1]].shape[0] != 1):
                    raise UnsupportedOperator(t, node.name, "non-contiguous constant indices: only positions 0…T−1 (or a slice of them) are translated")
                rows, width = shape_of(w)[0], shape_of(w)[-1]
                if start == 0 and n == rows:
                    env[node] = w
                else:
                    env[node] = b.emit("Slice", [w], node.name, {"axis": 0, "start": start, "end": start + n}, (n, width))
            elif t.startswith("aten.embedding"):
                w, idx = tensor_of(a[0], node.name), tensor_of(a[1], node.name)
                env[node] = b.emit("Gather", [w, idx], node.name, shape=shape_of(idx) + shape_of(w)[1:])
            elif t.startswith("aten.slice.Tensor"):
                x = tensor_of(a[0], node.name)
                sx = shape_of(x)
                dim = int(a[1]) if len(a) > 1 else 0
                dim = dim % len(sx)
                size = sx[dim]
                start = int(a[2]) if len(a) > 2 and a[2] is not None else 0
                end = int(a[3]) if len(a) > 3 and a[3] is not None else size
                step = int(a[4]) if len(a) > 4 else 1
                if step != 1:
                    raise UnsupportedOperator(t, node.name, f"step {step}: only contiguous slices are translated")
                start = max(0, start + size if start < 0 else start)
                end = min(size, end + size if end < 0 else end)
                if end <= start:
                    env[node] = ("empty", dim)
                    continue
                if dim == 0:
                    raise UnsupportedOperator(t, node.name, "slice along the batch axis")
                if start == 0 and end == size:
                    env[node] = x
                else:
                    out = list(sx)
                    out[dim] = end - start
                    env[node] = b.emit("Slice", [x], node.name, {"axis": dim, "start": start, "end": end}, tuple(out))
            elif t.startswith(("aten.dropout", "aten.native_dropout")):
                x = tensor_of(a[0], node.name)
                p = float(a[1])
                train = bool(a[2]) if len(a) > 2 else False
                out = b.emit("Dropout", [x], node.name, {"rate": p if train else 0.0}, shape_of(x))
                env[node] = ("multi", out) if "native" in t else out
            elif t.startswith(("aten.add.", "aten.sub.", "aten.mul.", "aten.div.")) and t.split(".")[2] in ("Tensor", "Scalar"):
                # A Python scalar or a uniform constant tensor becomes a frozen parameter.
                x, y = tensor_of(a[0], node.name), tensor_of(a[1], node.name)
                if len(a) > 2 or "alpha" in kw or kw.get("rounding_mode"):
                    raise UnsupportedOperator(t, node.name, "alpha factor or rounding division")
                op = {"add": "Add", "sub": "Sub", "mul": "Mul", "div": "Div"}[t.split(".")[1]]
                env[node] = b.emit(op, [x, y], node.name, shape=broadcast_shape(shape_of(x), shape_of(y)))
            elif t.startswith("aten.rsub."):
                # rsub(x, c) = c − x
                x, c = tensor_of(a[0], node.name), tensor_of(a[1], node.name)
                env[node] = b.emit("Sub", [c, x], node.name, shape=shape_of(x))
            elif t.startswith("aten.neg."):
                x = tensor_of(a[0], node.name)
                env[node] = b.emit("Mul", [x, const_param(-1.0, node.name)], node.name, shape=shape_of(x))
            elif t.startswith("aten.masked_fill."):
                # Only the causal mask is translated: a constant mask that is exactly the strict
                # triangle of future positions, filled with a "−∞" value.
                x = tensor_of(a[0], node.name)
                mask_node, fill = a[1], a[2]
                fill = uniform_value(fill) if not isinstance(fill, (int, float)) else float(fill)
                env[node] = causal_from(x, mask_node, fill, masked_where_true=True, ctx=node.name)
            elif t.startswith("aten.where."):
                # where(cond, a, b): keeps a where cond is true. One of the two is the mask value.
                cond, a1, b1 = a[0], a[1], a[2]
                a_fill = a1 if isinstance(a1, (int, float)) else (uniform_value(a1) if a1 in const_nodes else None)
                b_fill = b1 if isinstance(b1, (int, float)) else (uniform_value(b1) if b1 in const_nodes else None)
                if b_fill is not None and a_fill is None:
                    env[node] = causal_from(tensor_of(a1, node.name), cond, float(b_fill), masked_where_true=False, ctx=node.name)
                elif a_fill is not None and b_fill is None:
                    env[node] = causal_from(tensor_of(b1, node.name), cond, float(a_fill), masked_where_true=True, ctx=node.name)
                else:
                    raise UnsupportedOperator(t, node.name, "only the causal mask (where with a −∞ value) is translated")
            elif t.startswith("aten.scaled_dot_product_attention"):
                q, k, v = (tensor_of(a[i], node.name) for i in range(3))
                attn_mask = a[3] if len(a) > 3 else kw.get("attn_mask")
                dropout_p = a[4] if len(a) > 4 else kw.get("dropout_p", 0.0)
                is_causal = a[5] if len(a) > 5 else kw.get("is_causal", False)
                if dropout_p:
                    raise UnsupportedOperator(t, node.name, f"dropout {dropout_p} in attention")
                sq, sk, sv = shape_of(q), shape_of(k), shape_of(v)
                additive = None
                if attn_mask is not None and attn_mask not in const_nodes and isinstance(env.get(attn_mask), str):
                    # Additive mask computed in the graph (filled from an input mask): added to the
                    # scores, as PyTorch does.
                    additive = env[attn_mask]
                    attn_mask = None
                if attn_mask is not None and mask_is_empty(attn_mask):
                    # Constant mask that masks nothing (all true, or additive zero): no effect.
                    attn_mask = None
                if attn_mask is not None:
                    # An explicit mask is translated only if it is the causal mask.
                    if is_causal:
                        raise UnsupportedOperator(t, node.name, "both attn_mask and is_causal")
                    if sq[-2] != sk[-2]:
                        raise UnsupportedOperator(t, node.name, "mask between sequences of different lengths")
                    is_causal = attention_mask_is_causal(attn_mask, sq[-2], node.name)
                gqa = a[7] if len(a) > 7 else kw.get("enable_gqa", False)
                if sk[-3] != sq[-3]:
                    # Grouped-query attention: each key head serves h/h_kv queries.
                    if not gqa or sq[-3] % sk[-3]:
                        raise UnsupportedOperator(t, node.name, f"{sq[-3]} query heads for {sk[-3]} key heads")
                    n = sq[-3] // sk[-3]
                    k = repeat_heads(k, len(sk) - 3, n, f"{node.name}_k")
                    v = repeat_heads(v, len(sv) - 3, n, f"{node.name}_v")
                    sk, sv = shape_of(k), shape_of(v)
                r = len(sq)
                perm = list(range(r))
                perm[-1], perm[-2] = perm[-2], perm[-1]
                kt = b.emit("Transpose", [k], f"{node.name}_kT", {"perm": perm}, tuple(sk[p] for p in perm))
                scores_shape = sq[:-1] + (sk[-2],)
                s = b.emit("MatMul", [q, kt], f"{node.name}_scores", shape=scores_shape)
                scale = kw.get("scale")
                scale = float(scale) if scale is not None else 1.0 / float(np.sqrt(sq[-1]))
                s = b.emit("Mul", [s, const_param(scale, node.name)], f"{node.name}_scaled", shape=scores_shape)
                if additive is not None:
                    s = b.emit("Add", [s, additive], f"{node.name}_mask", shape=scores_shape)
                if is_causal:
                    if sq[-2] != sk[-2]:
                        raise UnsupportedOperator(t, node.name, "causal attention between sequences of different lengths")
                    s = b.emit("CausalMask", [s], f"{node.name}_masked", shape=scores_shape)
                w = b.emit("Softmax", [s], f"{node.name}_weights", {"axis": r - 1}, scores_shape)
                env[node] = b.emit("MatMul", [w, v], node.name, shape=scores_shape[:-1] + (sv[-1],))
            elif t.startswith(("aten.clone", "aten.detach", "aten.alias", "aten.contiguous", "aten._to_copy", "aten.to.")):
                env[node] = tensor_of(a[0], node.name)
            else:
                raise UnsupportedOperator(t, node.name)
        except (UnsupportedOperator, ExportError, KeyError, IndexError, TypeError, ValueError) as e:
            if _missing is None:
                if isinstance(e, (UnsupportedOperator, ExportError)):
                    raise
                raise ExportError(f"node '{node.name}' ({t}): {e}") from e
            # Coverage analysis: record what is missing and continue with a placeholder tensor of
            # the right shape, to see everything that is missing in one go.
            _missing.append((t, str(e).split(". Profile operators")[0][:240]))
            val = node.meta.get("val")
            vals = list(val) if isinstance(val, (list, tuple)) else [val]
            names = []
            for k, v in enumerate(vals):
                fake = _clean_name(f"missing.{node.name}.{k}", b.taken)
                b.shapes[fake] = tuple(int(d) for d in getattr(v, "shape", ()))
                names.append(fake)
            env[node] = ("multi", names[0]) if isinstance(val, (list, tuple)) else names[0]

    if _missing is not None:
        return None, None
    # Safeguard: every shape the driver computed must be the one PyTorch measured, or broadcast to
    # it (an embedding row shared by the whole batch). Otherwise a Reshape emitted further on would
    # carry a wrong shape: better a rejection here than a wrong computation elsewhere.
    for fx_node in gm.graph.nodes:
        gg_name = env.get(fx_node)
        val = fx_node.meta.get("val")
        if not isinstance(gg_name, str) or gg_name not in b.shapes or not hasattr(val, "shape"):
            continue
        mine, true = tuple(b.shapes[gg_name]), tuple(int(d) for d in val.shape)
        if mine != true and (len(mine) > len(true) or broadcast_shape(mine, true) != true):
            raise ExportError(
                f"node '{fx_node.name}': computed shape {mine}, PyTorch measures {true} (driver bug)"
            )
    if len(outputs) != 1:
        raise ExportError(f"the module returns {len(outputs)} outputs, the driver expects one")
    prediction = outputs[0]
    pred_shape = shape_of(prediction)
    if loss == "cross_entropy":
        tgt_shape = list(target_shape) if target_shape is not None else [0]
        target = {"name": "y", "shape": tgt_shape, "dtype": target_dtype or "i64"}
        loss_spec = {"cross_entropy": {"prediction": prediction, "target": target}}
    else:
        tgt_shape = list(target_shape) if target_shape is not None else [0] + [int(d) for d in pred_shape[1:]]
        target = {"name": "y", "shape": tgt_shape, "dtype": target_dtype or "f32"}
        loss_spec = {"mean_squared_error": {"prediction": prediction, "target": target}}

    description = {
        "format": FORMAT,
        "name": _NAME_RE.sub("-", name)[:64] or "model",
        "inputs": inputs_spec,
        "parameters": parameters,
        "nodes": b.nodes,
        "loss": loss_spec,
    }
    return description, weights


def coverage(model, example_inputs, *, input_dtypes: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """What the profile lacks to capture `model`, in one pass: each untranslated operator, with
    the number of nodes and the reason. Captures nothing; used to decide what to add.

    Returns `{"captured": bool, "missing": {op: {"count": n, "reason": ...}}, "export_error": str|None}`.
    """
    missing: List[Tuple[str, str]] = []
    try:
        export_model(model, example_inputs, "mse", input_dtypes=input_dtypes, _missing=missing)
    except Exception as e:  # noqa: BLE001 — torch.export can raise anything
        # torch.export itself refused (data-dependent control flow, for example).
        return {"captured": False, "missing": {}, "export_error": f"{type(e).__name__}: " + str(e).splitlines()[0][:300]}
    table: Dict[str, Any] = {}
    for op, reason in missing:
        entry = table.setdefault(op, {"count": 0, "reason": reason})
        entry["count"] += 1
    return {"captured": not table, "missing": table, "export_error": None}


def _node_values(ep, gm, example_inputs) -> Dict[Any, Any]:
    """Concrete value of each node of the captured graph, on the example inputs.

    Used to recognize what a constant subgraph actually computes (a causal mask, a scalar) instead
    of guessing from the shape of the code that produced it.
    """
    import torch
    from torch.fx import Interpreter

    sig = ep.graph_signature
    params = dict(sig.inputs_to_parameters)
    buffers = dict(sig.inputs_to_buffers)
    lifted = dict(getattr(sig, "inputs_to_lifted_tensor_constants", {}))
    consts = dict(getattr(ep, "constants", {}))
    args = []
    ui = 0
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        if node.name in params:
            args.append(ep.state_dict[params[node.name]])
        elif node.name in buffers:
            key = buffers[node.name]
            args.append(ep.state_dict[key] if key in ep.state_dict else consts[key])
        elif node.name in lifted:
            args.append(consts[lifted[node.name]])
        else:
            args.append(example_inputs[ui])
            ui += 1
    values: Dict[Any, Any] = {}

    class Recorder(Interpreter):
        def run_node(self, n):
            out = super().run_node(n)
            values[n] = out
            return out

    with torch.no_grad():
        Recorder(gm).run(*args)
    return values


def _constant_cone(gm, param_placeholders: set, user_placeholders: set) -> set:
    """Nodes that depend on neither an input nor a parameter: buffers, constants, and everything
    computed from them. They are not emitted; their value is read when needed."""
    from torch.fx import Node

    const: set = set()
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if node.name not in param_placeholders and node.name not in user_placeholders:
                const.add(node)
        elif node.op == "get_attr":
            const.add(node)
        elif node.op == "call_function":
            deps = [x for x in node.all_input_nodes if isinstance(x, Node)]
            if all(x in const for x in deps):
                const.add(node)
    return const


def _input_cone(gm, user_placeholders: set) -> set:
    """Nodes that depend on at least one input of the module: those that carry the batch axis."""
    cone: set = set()
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if node.name in user_placeholders:
                cone.add(node)
        elif node.op == "call_function" and any(x in cone for x in node.all_input_nodes):
            cone.add(node)
    return cone


def constant_parameter_names(description: Dict[str, Any]) -> List[str]:
    """The driver's synthetic parameters, absent from the original module's parameters: the model's
    scalars and position tables, all frozen."""
    synthetic = ("constant",) + POSITION_TABLES
    return [
        p["name"]
        for p in description.get("parameters", [])
        if (isinstance(p.get("init"), dict) and any(k in p["init"] for k in synthetic))
        # The frozen zeros a broadcast over the batch adds (`zeros_param`).
        or (p["name"].startswith("zeros.") and p.get("init") == "zeros" and p.get("trainable") is False)
    ]


def tied_parameters(model) -> Dict[str, str]:
    """Each parameter name of `model` -> the first name, in `named_parameters` order, of the tensor
    it shares storage with (itself when it shares nothing). Tied weights (a language-model head
    that is the embedding matrix) thus have one name: one gg parameter."""
    out: Dict[str, str] = {}
    first: Dict[Any, str] = {}
    if not hasattr(model, "named_parameters"):
        return out
    for key, p in model.named_parameters(remove_duplicate=False):
        if p.numel() == 0:
            ident = ("object", id(p))
        else:
            ident = (p.untyped_storage().data_ptr(), p.storage_offset(), tuple(p.shape), tuple(p.stride()))
        out[key] = first.setdefault(ident, key)
    return out


def state_dict_from_bundle(model, bundle: Dict[str, np.ndarray], ignore: Sequence[str] = ()):
    """Load a GGB1 checkpoint (gg names) into a PyTorch module, matching by name.

    `ignore` lists the synthetic parameters (model constants) that do not exist in the module:
    see `constant_parameter_names`."""
    import torch

    sd = model.state_dict()
    # Only parameters are trained and transported; buffers (a mask, running statistics) stay those
    # of the module. The gg names are recomputed in capture order.
    # Tied weights (the language-model head that shares the embedding matrix, in Mamba, GPT-2,
    # Gemma, T5…) are one gg parameter, named after the first alias (`tied_parameters`), exactly as
    # the capture names them; loading it into that alias loads it into all of them.
    tied = tied_parameters(model)
    aliases = list(model.named_parameters(remove_duplicate=False))
    trained = [k for k, _ in model.named_parameters()]
    owner = {k: id(p) for k, p in aliases}
    taken: set = set()
    mapping = {}
    for key, _ in aliases:
        if tied.get(key, key) == key:
            mapping[_clean_name(key, taken)] = key
    new_sd = {}
    skip = set(ignore)
    for gg_name, arr in bundle.items():
        if gg_name in skip:
            continue
        key = mapping.get(gg_name)
        if key is None:
            raise ExportError(
                f"the checkpoint contains '{gg_name}', which the module does not have (a checkpoint "
                "of a description captured before tied weights were one parameter holds each alias "
                "separately: recapture the module)"
            )
        new_sd[key] = torch.from_numpy(np.ascontiguousarray(arr)).reshape(sd[key].shape).to(sd[key].dtype)
    covered = {owner[k] for k in new_sd if k in owner}
    missing = [k for k in trained if owner[k] not in covered]
    if missing:
        raise ExportError(f"the checkpoint does not cover: {missing[:5]}")
    # strict=False: buffers absent from the checkpoint keep their value; the parameters, however,
    # are all covered, as checked just above.
    model.load_state_dict(new_sd, strict=False)
    return model
