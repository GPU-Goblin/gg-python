"""With a lab node (ADR-0041), the library sends the datasets and starting weights to the node,
creates the run there and reads its progress and trained weights from it; the control plane only
receives the model's description. Checked against two local fake services."""
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from gpugoblin.client import ApiError, Client
from gpugoblin.tensors import encode_bundle


def fake(routes):
    """A local service answering `routes[(method, path)](body) -> (status, bytes, content-type)`,
    recording every request."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def _answer(self):
            n = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(n) if n else b""
            path = self.path.split("?")[0]
            seen.append((self.command, path, dict(self.headers), body))
            handler = routes.get((self.command, path))
            status, payload, ctype = handler(body) if handler else (404, b"{}", "application/json")
            self.send_response(status)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = do_PUT = _answer

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, seen


def as_json(v):
    return 200, json.dumps(v).encode(), "application/json"


def test_datasets_runs_and_weights_go_through_the_node():
    weights = {"w": np.arange(6, dtype=np.float32).reshape(2, 3)}
    bundle = encode_bundle(weights)
    ref = {"hash": hashlib.sha256(bundle).hexdigest(), "size": len(bundle)}
    view = {"run_id": "r1", "state": "completed", "checkpoint": {"weights": {"checkpoint": ref}}}

    server, server_seen = fake({("GET", "/v1/info"): lambda b: as_json({"network_id": "gg-lab"})})
    node, node_seen = fake({
        ("PUT", "/v1/projects/p1/data"): lambda b: as_json(
            {"hash": hashlib.sha256(b).hexdigest(), "size": len(b)}
        ),
        ("POST", "/v1/projects/p1/runs"): lambda b: as_json({"run_id": "r1", "echo": json.loads(b)}),
        ("GET", "/v1/runs/r1"): lambda b: as_json(
            {"run": view, "current_round_tasks": {"queued": 2}, "checkpoint_on_this_node": True}
        ),
        ("GET", "/v1/runs/r1/checkpoint/checkpoint"): lambda b: (200, bundle, "application/octet-stream"),
    })
    try:
        c = Client(f"http://127.0.0.1:{server.server_port}", "ggt_lab", node=f"http://127.0.0.1:{node.server_port}")
        data = c.upload_data("p1", b"\x01" * 100, "ggt1-f32")
        assert data == {"hash": hashlib.sha256(b"\x01" * 100).hexdigest(), "size": 100}
        run = c.create_graph_run("p1", {"hash": "a" * 64, "size": 1}, {"x": data},
                                 batch_per_round=8, tasks_per_round=2, rounds=1)
        assert run["run_id"] == "r1"
        assert c.run("r1")["state"] == "completed"
        assert c.node_run("r1")["current_round_tasks"] == {"queued": 2}
        assert c.download_checkpoint_part("r1", "checkpoint", ref) == bundle
    finally:
        server.shutdown()
        node.shutdown()
    # The node received the data, the run, the progress and weight requests, each with the
    # researcher's token and the network of the server this client talks to.
    assert [(m, p) for m, p, _, _ in node_seen] == [
        ("PUT", "/v1/projects/p1/data"),
        ("POST", "/v1/projects/p1/runs"),
        ("GET", "/v1/runs/r1"),
        ("GET", "/v1/runs/r1"),
        ("GET", "/v1/runs/r1/checkpoint/checkpoint"),
    ]
    for _, _, headers, _ in node_seen:
        assert headers["Authorization"] == "Bearer ggt_lab"
        assert headers["X-GG-Network"] == "gg-lab"
    # The control plane was only asked who it is: no data went there.
    assert [(m, p) for m, p, _, _ in server_seen] == [("GET", "/v1/info")]


def test_weights_that_do_not_match_their_reference_are_refused():
    server, _ = fake({("GET", "/v1/info"): lambda b: as_json({"network_id": "gg-lab"})})
    node, _ = fake({
        ("GET", "/v1/runs/r1/checkpoint/checkpoint"): lambda b: (200, b"tampered", "application/octet-stream"),
        ("PUT", "/v1/projects/p1/data"): lambda b: as_json({"hash": "0" * 64, "size": len(b)}),
    })
    try:
        c = Client(f"http://127.0.0.1:{server.server_port}", "ggt_lab", node=f"http://127.0.0.1:{node.server_port}")
        with pytest.raises(ApiError):
            c.download_checkpoint_part("r1", "checkpoint", {"hash": "1" * 64, "size": 8})
        with pytest.raises(ApiError):
            c.upload_data("p1", b"abc", "ggt1-f32")
    finally:
        server.shutdown()
        node.shutdown()


def test_without_a_node_nothing_changes():
    c = Client("http://127.0.0.1:9", "ggt_x")
    assert c.node is None
    with pytest.raises(RuntimeError):
        c.node_status()
