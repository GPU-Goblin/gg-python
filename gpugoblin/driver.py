"""Driver: from a PyTorch module to collective training on the grid, and back."""

from __future__ import annotations

import json
import warnings
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .client import Client
from .export import constant_parameter_names, export_model, state_dict_from_bundle
from .tensors import decode_bundle, encode_bundle, encode_tensor

#: Above this size, initial weights are uploaded one parameter at a time and the server assembles
#: the checkpoint shard by shard (gg-server/docs/ROADMAP.md, P11 C2): no single upload carries the whole model.
SINGLE_UPLOAD_BYTES = 256 << 20


class PlanError(RuntimeError):
    """The grid cannot run this training as requested; the message says what to do."""


class Trainer:
    """Capture a module, upload its description, weights and data, and follow the run."""

    def __init__(
        self,
        client: Client,
        project: str,
        model,
        loss: str,
        example_input,
        *,
        name: str = "model",
        input_dtypes=None,
        target_shape=None,
    ):
        self.client = client
        self.project = project
        self.model = model
        self.description, self.weights = export_model(
            model, example_input, loss, name=name, input_dtypes=input_dtypes, target_shape=target_shape
        )
        self.graph_ref: Optional[Dict[str, Any]] = None
        self.checkpoint_ref: Optional[Dict[str, Any]] = None
        self.parameter_refs: Optional[Dict[str, Dict[str, Any]]] = None

    @property
    def parameter_count(self) -> int:
        return int(sum(int(np.prod(p["shape"])) for p in self.description["parameters"]))

    def description_json(self) -> str:
        return json.dumps(self.description, indent=2, ensure_ascii=False)

    @property
    def checkpoint_bytes(self) -> int:
        return int(sum(4 * int(np.prod(np.shape(w))) for w in self.weights.values()))

    def upload(self) -> None:
        """Upload the description and the initial weights (idempotent by hash): one bundle if the
        model is small, one tensor per parameter otherwise."""
        self.graph_ref = self.client.upload_artifact(
            self.project, self.description_json().encode("utf-8"), "gg-graph-v1"
        )
        if self.checkpoint_bytes <= SINGLE_UPLOAD_BYTES:
            self.checkpoint_ref = self.client.upload_artifact(
                self.project, encode_bundle(self.weights), "ggb1-f32"
            )
        else:
            shapes = {p["name"]: p["shape"] for p in self.description["parameters"]}
            self.parameter_refs = {
                name: self.client.upload_artifact(
                    self.project,
                    encode_tensor(np.asarray(w, dtype=np.float32).reshape(shapes[name])),
                    "ggt1-f32",
                )
                for name, w in self.weights.items()
            }

    def fit(
        self,
        x,
        y,
        *,
        rounds: int,
        batch_per_round: int,
        tasks_per_round: int,
        lr: float = 1e-3,
        optimizer: str = "adam",
        seed: int = 1,
        max_round_retries: int = 2,
        extra_inputs: Optional[Mapping[str, Any]] = None,
        shard_bytes: Optional[int] = None,
        segments: Optional[int] = None,
        offload_parameters: Optional[bool] = None,
        stage_memory_bytes: Optional[int] = None,
        tile_max_param_bytes: Optional[int] = None,
        gradient_precision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create the run. `x` feeds the first input of the graph, `y` the target; additional inputs
        (by name) come from `extra_inputs`. `gradient_precision`: `"bf16"` (the grid's default, half
        the bytes per task) or `"f32"`; weights and optimizer state stay f32 on the server either
        way."""
        if gradient_precision not in (None, "bf16", "f32"):
            raise ValueError(f"gradient_precision {gradient_precision!r}: expected 'bf16' or 'f32'")
        if segments is None and offload_parameters is None and stage_memory_bytes is None:
            # Memory levers chosen by the grid, based on what its workers advertise (C4).
            plan = self.plan(batch_per_round, tasks_per_round)
            rec = plan.get("recommendation")
            if rec is not None:
                segments, offload_parameters = rec["segments"], rec["offload_parameters"]
            elif plan.get("staged") is not None and stage_memory_bytes is None:
                # No machine can hold the whole model: it runs in stages (gg-server/docs/ROADMAP.md, D2), after
                # tiling any matrix too large for one machine if needed (D3).
                stage_memory_bytes = plan["staged"]["stage_memory_bytes"]
                tile_max_param_bytes = plan["staged"].get("tile_max_param_bytes")
            elif plan.get("workers_seen", 0) == 0:
                warnings.warn(f"{plan.get('advice')}; submitted without a memory lever")
            else:
                raise PlanError(plan.get("advice", "no worker can hold this task"))
        segments = 1 if segments is None else int(segments)
        offload_parameters = bool(offload_parameters)
        if self.graph_ref is None or (self.checkpoint_ref is None and self.parameter_refs is None):
            self.upload()
        inputs = self.description["inputs"]
        arrays = {inputs[0]["name"]: np.asarray(x)}
        for spec in inputs[1:]:
            if not extra_inputs or spec["name"] not in extra_inputs:
                raise ValueError(f"graph input '{spec['name']}' has no dataset (extra_inputs)")
            arrays[spec["name"]] = np.asarray(extra_inputs[spec["name"]])
        target_name = next(iter(self.description["loss"].values()))["target"]["name"]
        arrays[target_name] = np.asarray(y)
        # A dataset larger than one upload is sent as parts of consecutive rows; each task then
        # receives only the rows of its micro-batch (gg-server/docs/ROADMAP.md, P11 C4).
        datasets: Dict[str, Any] = {}
        dataset_parts: Dict[str, Any] = {}
        for name, a in arrays.items():
            a = np.asarray(a, dtype=np.float32)
            if a.nbytes <= SINGLE_UPLOAD_BYTES:
                datasets[name] = self.client.upload_artifact(self.project, encode_tensor(a), "ggt1-f32")
            else:
                row = max(1, a.nbytes // max(1, a.shape[0]))
                per = max(1, SINGLE_UPLOAD_BYTES // row)
                dataset_parts[name] = [
                    self.client.upload_artifact(self.project, encode_tensor(a[i:i + per]), "ggt1-f32")
                    for i in range(0, a.shape[0], per)
                ]
        return self.client.create_graph_run(
            self.project,
            self.graph_ref,
            datasets,
            dataset_parts=dataset_parts or None,
            name=self.description["name"],
            optimizer=optimizer,
            learning_rate=lr,
            batch_per_round=batch_per_round,
            tasks_per_round=tasks_per_round,
            rounds=rounds,
            seed=seed,
            max_round_retries=max_round_retries,
            initial_checkpoint=self.checkpoint_ref,
            initial_parameters=self.parameter_refs,
            shard_bytes=shard_bytes,
            segments=segments,
            offload_parameters=offload_parameters,
            stage_memory_bytes=stage_memory_bytes,
            tile_max_param_bytes=tile_max_param_bytes,
            gradient_precision=gradient_precision,
        )

    def plan(self, batch_per_round: int, tasks_per_round: int) -> Dict[str, Any]:
        """What the grid can do with this model before any data is sent: required profile, memory of
        one task, workers able to hold it, recommended levers, and a sentence of advice."""
        return self.client.plan(self.description, batch_per_round, tasks_per_round)

    def wait(self, run_id: str, poll: float = 2.0, timeout: Optional[float] = None) -> Dict[str, Any]:
        return self.client.wait_run(run_id, poll=poll, timeout=timeout)

    def checkpoint(self, run_id: str) -> Dict[str, np.ndarray]:
        """The run's current checkpoint, decoded: in one piece or reassembled from its shards."""
        view = self.client.run(run_id)
        out: Dict[str, np.ndarray] = {}
        for name, ref in sorted(view["checkpoint"]["weights"].items()):
            if not name.startswith("checkpoint"):
                raise ValueError(f"unexpected checkpoint part: {name}")
            out.update(decode_bundle(self.client.download_artifact(self.project, ref)))
        # Matrices tiled by the grid (D3): reassembled, in the reverse order of tiling.
        for t in reversed((view.get("config", {}).get("graph") or {}).get("tiling") or []):
            names = [f"{t['param']}.tile{i}of{t['tiles']}" for i in range(t["tiles"])]
            out[t["param"]] = np.concatenate([out.pop(n) for n in names], axis=int(t["axis"]))
        return out

    def load_checkpoint_into(self, model, run_id: str):
        """Load the current checkpoint into a PyTorch module (matched by name)."""
        return state_dict_from_bundle(
            model, self.checkpoint(run_id), ignore=constant_parameter_names(self.description)
        )
