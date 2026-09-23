"""Minimal REST client for the control plane's `/v1` API."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import requests

# Longest a request waits in total for the server to accept it again (429, 503), as the Rust client.
BACKPRESSURE_BUDGET_S = 60.0


def _retry_after(r: requests.Response, attempt: int) -> float:
    """How long the server asks to wait (`Retry-After`, in seconds), bounded; otherwise a short
    backoff growing with the attempt."""
    try:
        wait = float(int(r.headers.get("retry-after", "").strip()))
    except ValueError:
        wait = 0.2 * attempt
    return min(wait, 10.0)


class ApiError(RuntimeError):
    """Error response from the server, with its HTTP status and body."""

    def __init__(self, status: int, body: Any):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


class Client:
    """Access authenticated with a researcher API token."""

    def __init__(self, server: str, token: str, timeout: float = 120.0):
        self.base = server.rstrip("/") + "/v1"
        self.token = token
        self.timeout = timeout
        self._http = requests.Session()
        self._http.headers["Authorization"] = f"Bearer {token}"

    # -- low level -----------------------------------------------------------------------------

    def _send(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        """Send a request, and replay it while the server pushes back: 429 (rate limit,
        `Retry-After` honoured) or 503 (an aborted transaction, replayable as is). A push-back is
        not a failure: uploading the ~270 parameters of a staged 135 M model otherwise stopped
        at the end of the burst."""
        waited, attempt = 0.0, 0
        while True:
            attempt += 1
            r = self._http.request(method, url, **kwargs)
            if r.status_code in (429, 503) and waited < BACKPRESSURE_BUDGET_S:
                wait = _retry_after(r, attempt)
                time.sleep(wait)
                waited += wait
                continue
            return r

    def _check(self, r: requests.Response) -> Any:
        if r.status_code >= 400:
            try:
                body = r.json()
            except ValueError:
                body = r.text
            raise ApiError(r.status_code, body)
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        return r.content

    def get(self, path: str) -> Any:
        return self._check(self._send("GET", self.base + path, timeout=self.timeout))

    def post(self, path: str, body: Mapping[str, Any]) -> Any:
        return self._check(
            self._send("POST", self.base + path, json=dict(body), timeout=self.timeout)
        )

    # -- discovery ----------------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        return self.get("/info")

    def grid_capacity(self) -> Dict[str, Any]:
        """Authorized workers active recently and what they advertise: host memory, accelerator
        memory, operator profile. Used to plan before submitting."""
        return self.get("/researcher/grid/capacity")

    def plan(self, description: Mapping[str, Any], batch_per_round: int, tasks_per_round: int) -> Dict[str, Any]:
        """What the grid can do with this description, before anything is sent: required profile,
        memory of one task, workers able to hold it, and the cheapest memory levers."""
        return self.post("/researcher/plan", {
            "graph": dict(description),
            "batch_per_round": int(batch_per_round),
            "tasks_per_round": int(tasks_per_round),
        })

    def operator_profile(self) -> Dict[str, Any]:
        """The operator profile this server accepts (`GET /v1/catalog/ops`)."""
        return self.get("/catalog/ops")

    # -- artifacts ----------------------------------------------------------------------------

    @staticmethod
    def sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def upload_artifact(self, project: str, data: bytes, schema: str) -> Dict[str, Any]:
        """Upload bytes and return the reference `{hash, size}`. Idempotent by hash."""
        h = self.sha256(data)
        r = self._send(
            "PUT",
            f"{self.base}/researcher/projects/{project}/artifacts/{h}",
            params={"size": len(data), "schema": schema},
            data=data,
            headers={"Content-Type": "application/octet-stream"},
            timeout=max(self.timeout, 600.0),
        )
        self._check(r)
        return {"hash": h, "size": len(data)}

    def download_artifact(self, project: str, ref: Mapping[str, Any]) -> bytes:
        r = self._send(
            "GET",
            f"{self.base}/researcher/projects/{project}/artifacts/{ref['hash']}",
            timeout=max(self.timeout, 600.0),
        )
        data = self._check(r)
        if not isinstance(data, (bytes, bytearray)):
            raise ApiError(r.status_code, "non-binary response")
        if len(data) != int(ref["size"]) or self.sha256(data) != ref["hash"]:
            raise ApiError(r.status_code, "received artifact differs from the reference (size or hash)")
        return bytes(data)

    # -- collective training -----------------------------------------------------------------

    def create_graph_run(
        self,
        project: str,
        graph: Mapping[str, Any],
        datasets: Mapping[str, Mapping[str, Any]],
        *,
        name: str = "gpugoblin",
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        batch_per_round: int,
        tasks_per_round: int,
        rounds: int,
        seed: int = 1,
        max_round_retries: int = 2,
        priority: int = 0,
        initial_checkpoint: Optional[Mapping[str, Any]] = None,
        initial_parameters: Optional[Mapping[str, Mapping[str, Any]]] = None,
        dataset_parts: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
        shard_bytes: Optional[int] = None,
        segments: int = 1,
        offload_parameters: bool = False,
        stage_memory_bytes: Optional[int] = None,
        tile_max_param_bytes: Optional[int] = None,
        gradient_precision: Optional[str] = None,
    ) -> Dict[str, Any]:
        if gradient_precision not in (None, "bf16", "f32"):
            raise ValueError(f"gradient_precision {gradient_precision!r}: expected 'bf16' or 'f32'")
        body: Dict[str, Any] = {
            "name": name,
            "graph": dict(graph),
            "datasets": {k: dict(v) for k, v in datasets.items()},
            "optimizer": optimizer,
            "learning_rate": learning_rate,
            "batch_per_round": batch_per_round,
            "tasks_per_round": tasks_per_round,
            "rounds": rounds,
            "seed": seed,
            "max_round_retries": max_round_retries,
            "priority": priority,
        }
        if initial_checkpoint is not None:
            body["initial_checkpoint"] = dict(initial_checkpoint)
        if dataset_parts:
            body["dataset_parts"] = {k: [dict(r) for r in v] for k, v in dataset_parts.items()}
        if initial_parameters is not None:
            body["initial_parameters"] = {k: dict(v) for k, v in initial_parameters.items()}
        if shard_bytes is not None:
            body["shard_bytes"] = int(shard_bytes)
        if segments != 1:
            body["segments"] = int(segments)
        if offload_parameters:
            body["offload_parameters"] = True
        if stage_memory_bytes is not None:
            body["stage_memory_bytes"] = int(stage_memory_bytes)
        if tile_max_param_bytes is not None:
            body["tile_max_param_bytes"] = int(tile_max_param_bytes)
        if gradient_precision is not None:
            # bf16 (the grid's default) halves what each task sends; f32 keeps every bit (ADR-0035).
            body["gradient_precision"] = gradient_precision
        return self.post(f"/researcher/projects/{project}/collective", body)

    def run(self, run_id: str) -> Dict[str, Any]:
        return self.get(f"/researcher/collective/{run_id}")

    def cancel_run(self, run_id: str) -> None:
        self.post(f"/researcher/collective/{run_id}/cancel", {})

    def wait_run(self, run_id: str, poll: float = 2.0, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for a run to end (completed, failed or canceled)."""
        started = time.monotonic()
        while True:
            view = self.run(run_id)
            if view.get("state") in ("completed", "failed", "cancelled"):
                return view
            if timeout is not None and time.monotonic() - started > timeout:
                raise TimeoutError(f"run {run_id} still in progress after {timeout} s")
            time.sleep(poll)
