import os
import shutil
import subprocess

import numpy as np
import pytest

from gpugoblin import tensors
from gg_binary import GG  # noqa: E402


def test_ggt1_layout_and_roundtrip():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = tensors.encode_tensor(a)
    assert b[:4] == b"GGT1" and b[4] == 1 and b[5] == 2
    assert len(b) == 6 + 16 + 24
    assert b[6:14] == (2).to_bytes(8, "little")
    np.testing.assert_array_equal(tensors.decode_tensor(b), a)
    with pytest.raises(tensors.FormatError):
        tensors.decode_tensor(b + b"\0")
    with pytest.raises(tensors.FormatError):
        tensors.encode_tensor(np.array([np.nan], dtype=np.float32))


def test_ggb1_sorted_unique_and_roundtrip():
    entries = {"b": np.ones((2,), np.float32), "a": np.zeros((1, 2), np.float32)}
    raw = tensors.encode_bundle(entries)
    assert raw[:4] == b"GGB1"
    assert raw[8:10] == (1).to_bytes(2, "little") and raw[10:11] == b"a"
    back = tensors.decode_bundle(raw)
    assert list(back) == ["a", "b"]
    np.testing.assert_array_equal(back["a"], entries["a"])
    assert len(raw) == tensors.bundle_size({"a": (1, 2), "b": (2,)})
    # Same content, different insertion order: same bytes.
    assert tensors.encode_bundle({"a": entries["a"], "b": entries["b"]}) == raw


@pytest.mark.skipif(not GG.exists(), reason="gg binary not found (set GG_BIN)")
def test_rust_reads_python_bundle(tmp_path):
    entries = {"l0.wq": np.random.rand(4, 4).astype(np.float32), "emb": np.random.rand(3, 4).astype(np.float32)}
    p = tmp_path / "ck.ggb"
    tensors.write_bundle(p, entries)
    out = subprocess.run([str(GG), "data", "inspect", str(p), "--json"], capture_output=True, text=True, check=True)
    import json
    got = {e["name"]: tuple(e["shape"]) for e in json.loads(out.stdout)}
    assert got == {"emb": (3, 4), "l0.wq": (4, 4)}
