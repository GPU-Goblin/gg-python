"""GPU Goblin's bounded binary formats, in NumPy.

GGT1 (one f32 tensor):
    magic "GGT1" | u8 dtype (1 = f32 LE) | u8 ndim (1..=4) | u64 LE shape[ndim] | data

GGB1 (several named tensors, a checkpoint):
    magic "GGB1" | u32 LE count | count × (u16 LE name_len | name | u64 LE tensor_len) | count × GGT1

The entries of a GGB1 are sorted by name (UTF-8 byte order), so a set of tensors has exactly
one encoding: the same checkpoint has the same hash everywhere.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

MAGIC = b"GGT1"
BUNDLE_MAGIC = b"GGB1"
DTYPE_F32 = 1
MAX_NDIM = 4
MAX_BUNDLE_ENTRIES = 4096
MAX_BUNDLE_NAME = 64


class FormatError(ValueError):
    """Malformed GGT1/GGB1 file."""


def encode_tensor(array) -> bytes:
    """Encode an array (converted to contiguous f32) in the GGT1 format."""
    a = np.ascontiguousarray(np.asarray(array), dtype="<f4")
    if a.ndim == 0 or a.ndim > MAX_NDIM:
        raise FormatError(f"rank {a.ndim} outside 1..={MAX_NDIM}")
    if any(d == 0 for d in a.shape):
        raise FormatError(f"shape {a.shape} has a zero dimension")
    if not np.all(np.isfinite(a)):
        raise FormatError("non-finite value")
    header = MAGIC + bytes([DTYPE_F32, a.ndim]) + b"".join(struct.pack("<Q", d) for d in a.shape)
    return header + a.tobytes()


def decode_tensor(data: bytes) -> np.ndarray:
    """Decode a GGT1; the buffer must be consumed exactly."""
    if len(data) < 6 or data[:4] != MAGIC:
        raise FormatError("expected GGT1 magic")
    if data[4] != DTYPE_F32:
        raise FormatError(f"unsupported dtype {data[4]}")
    ndim = data[5]
    if ndim == 0 or ndim > MAX_NDIM:
        raise FormatError(f"rank {ndim} outside 1..={MAX_NDIM}")
    header = 6 + 8 * ndim
    if len(data) < header:
        raise FormatError("truncated header")
    shape = struct.unpack("<" + "Q" * ndim, data[6:header])
    count = 1
    for d in shape:
        if d == 0:
            raise FormatError("zero dimension")
        count *= d
    if len(data) != header + 4 * count:
        raise FormatError(f"size {len(data)}, expected {header + 4 * count}")
    return np.frombuffer(data, dtype="<f4", offset=header).reshape(shape).copy()


def tensor_size(shape) -> int:
    """Exact size of a GGT1 with this shape."""
    n = 1
    for d in shape:
        n *= int(d)
    return 6 + 8 * len(shape) + 4 * n


def encode_bundle(entries: Mapping[str, "np.ndarray"]) -> bytes:
    """Encode named tensors in the GGB1 format (entries sorted by name)."""
    if len(entries) > MAX_BUNDLE_ENTRIES:
        raise FormatError(f"{len(entries)} entries, maximum {MAX_BUNDLE_ENTRIES}")
    names = sorted(entries.keys(), key=lambda s: s.encode("utf-8"))
    index = bytearray()
    payload = bytearray()
    for name in names:
        raw = name.encode("utf-8")
        if not raw or len(raw) > MAX_BUNDLE_NAME:
            raise FormatError(f"name '{name}' has length {len(raw)}, outside 1..={MAX_BUNDLE_NAME}")
        t = encode_tensor(entries[name])
        index += struct.pack("<H", len(raw)) + raw + struct.pack("<Q", len(t))
        payload += t
    return BUNDLE_MAGIC + struct.pack("<I", len(names)) + bytes(index) + bytes(payload)


def decode_bundle(data: bytes) -> Dict[str, np.ndarray]:
    """Decode a GGB1."""
    if len(data) < 8 or data[:4] != BUNDLE_MAGIC:
        raise FormatError("expected GGB1 magic")
    (count,) = struct.unpack("<I", data[4:8])
    if count > MAX_BUNDLE_ENTRIES:
        raise FormatError(f"{count} entries, maximum {MAX_BUNDLE_ENTRIES}")
    pos = 8
    index = []
    previous = None
    for _ in range(count):
        (name_len,) = struct.unpack("<H", data[pos : pos + 2])
        pos += 2
        raw = data[pos : pos + name_len]
        pos += name_len
        (tensor_len,) = struct.unpack("<Q", data[pos : pos + 8])
        pos += 8
        if previous is not None and raw <= previous:
            raise FormatError(f"unsorted index: {raw!r} after {previous!r}")
        previous = raw
        index.append((raw.decode("utf-8"), tensor_len))
    out: Dict[str, np.ndarray] = {}
    for name, tensor_len in index:
        out[name] = decode_tensor(data[pos : pos + tensor_len])
        pos += tensor_len
    if pos != len(data):
        raise FormatError(f"{len(data) - pos} trailing byte(s)")
    return out


def bundle_size(shapes: Mapping[str, tuple]) -> int:
    """Exact size of a GGB1 for named shapes."""
    total = 8
    for name, shape in shapes.items():
        total += 2 + len(name.encode("utf-8")) + 8 + tensor_size(shape)
    return total


def write_tensor(path, array) -> None:
    Path(path).write_bytes(encode_tensor(array))


def read_tensor(path) -> np.ndarray:
    return decode_tensor(Path(path).read_bytes())


def write_bundle(path, entries) -> None:
    Path(path).write_bytes(encode_bundle(entries))


def read_bundle(path) -> Dict[str, np.ndarray]:
    return decode_bundle(Path(path).read_bytes())
