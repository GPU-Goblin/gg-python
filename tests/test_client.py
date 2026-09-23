"""The client replays a request while the server pushes back (429, 503), as the Rust client does."""
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import gpugoblin.client as client_mod
from gpugoblin.client import ApiError, Client


def serve(statuses):
    """A local server answering the given statuses in turn (the last one repeats), counting calls."""
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def _answer(self):
            n = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(n) if n else b""
            calls.append((self.command, body))
            status = statuses[min(len(calls), len(statuses)) - 1]
            payload = b'{"ok": true}' if status == 200 else b"rate limit exceeded"
            self.send_response(status)
            if status == 429:
                self.send_header("retry-after", "1")
            self.send_header("content-type", "application/json" if status == 200 else "text/plain")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = do_PUT = _answer

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, calls


def test_a_rate_limited_upload_is_replayed_after_retry_after():
    httpd, calls = serve([429, 503, 200])
    try:
        c = Client(f"http://127.0.0.1:{httpd.server_port}", "ggt_test")
        t = time.monotonic()
        ref = c.upload_artifact("p", b"\x01\x02\x03", "ggt1-f32")
        elapsed = time.monotonic() - t
    finally:
        httpd.shutdown()
    assert ref["size"] == 3
    # The same bytes each time, and the server's Retry-After (1 s) honoured.
    assert [m for m, _ in calls] == ["PUT", "PUT", "PUT"]
    assert all(b == b"\x01\x02\x03" for _, b in calls)
    assert elapsed >= 1.0


def test_a_push_back_longer_than_the_budget_is_reported(monkeypatch):
    monkeypatch.setattr(client_mod, "BACKPRESSURE_BUDGET_S", 1.5)
    httpd, calls = serve([429])
    try:
        c = Client(f"http://127.0.0.1:{httpd.server_port}", "ggt_test")
        with pytest.raises(ApiError) as e:
            c.post("/researcher/plan", {})
    finally:
        httpd.shutdown()
    assert e.value.status == 429
    assert len(calls) == 3  # waits of 1 s until the budget is spent, then the error
