"""GPU Goblin: Python client and PyTorch driver.

The researcher writes ordinary PyTorch; the driver captures the graph, describes it as
`gg-graph/v1`, checks it against the published operator profile and submits it as a collective
training run. What crosses the network is a description and tensors, never a Python function.
"""

from .tensors import (
    decode_bundle,
    decode_tensor,
    encode_bundle,
    encode_tensor,
    read_bundle,
    read_tensor,
    write_bundle,
    write_tensor,
)
from .client import Client, ApiError
from .export import (
    PROFILE_ONNX_EQUIVALENTS,
    ExportError,
    UnsupportedOperator,
    batched_experts,
    coverage,
    export_model,
    freeze_rotary_embeddings,
)
from .driver import PlanError, Trainer
from .formula import FormulaError, formula

__all__ = [
    "ApiError",
    "Client",
    "ExportError",
    "FormulaError",
    "PROFILE_ONNX_EQUIVALENTS",
    "PlanError",
    "Trainer",
    "UnsupportedOperator",
    "batched_experts",
    "decode_bundle",
    "decode_tensor",
    "encode_bundle",
    "encode_tensor",
    "coverage",
    "export_model",
    "formula",
    "freeze_rotary_embeddings",
    "read_bundle",
    "read_tensor",
    "write_bundle",
    "write_tensor",
]
__version__ = "0.1.0"
