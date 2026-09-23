"""Driver and sharded checkpoint (gg-server/docs/ROADMAP.md, P11 C2), without a server: a fake client records
what the driver sends and returns what a server would return."""

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

import gpugoblin.driver as driver  # noqa: E402
from gpugoblin import Trainer, tensors  # noqa: E402


class FakeClient:
    def __init__(self):
        self.store = {}
        self.runs = {}
        self.created = None

    def upload_artifact(self, project, data, schema):
        h = hashlib.sha256(data).hexdigest()
        self.store[h] = data
        return {"hash": h, "size": len(data)}

    def download_artifact(self, project, ref):
        return self.store[ref["hash"]]

    plan_answer = {"recommendation": None, "workers_seen": 0, "advice": "no active worker"}

    def plan(self, description, batch_per_round, tasks_per_round):
        return self.plan_answer

    def create_graph_run(self, project, graph, datasets, **kw):
        self.created = kw
        self.datasets = datasets
        return {"run_id": "r1"}

    def run(self, run_id):
        return self.runs[run_id]


def model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 16), nn.GELU(), nn.Linear(16, 3))


def test_small_models_send_one_bundle_large_ones_one_tensor_per_parameter(monkeypatch):
    x = np.random.default_rng(0).standard_normal((32, 6)).astype(np.float32)
    y = np.zeros(32, dtype=np.float32)
    c = FakeClient()
    t = Trainer(c, "p", model(), "cross_entropy", torch.from_numpy(x[:2]))
    with pytest.warns(UserWarning, match="no active worker"):
        t.fit(x, y, rounds=1, batch_per_round=8, tasks_per_round=2)
    assert c.created["initial_checkpoint"] is not None
    assert c.created["initial_parameters"] is None

    monkeypatch.setattr(driver, "SINGLE_UPLOAD_BYTES", 0)
    c = FakeClient()
    t = Trainer(c, "p", model(), "cross_entropy", torch.from_numpy(x[:2]))
    t.fit(x, y, rounds=1, batch_per_round=8, tasks_per_round=2, shard_bytes=4096, segments=2)
    refs = c.created["initial_parameters"]
    assert c.created["initial_checkpoint"] is None
    assert set(refs) == {p["name"] for p in t.description["parameters"]}
    assert c.created["shard_bytes"] == 4096 and c.created["segments"] == 2
    # Each uploaded tensor has the shape declared by the description and the module's weights.
    for p in t.description["parameters"]:
        got = tensors.decode_tensor(c.store[refs[p["name"]]["hash"]])
        assert list(got.shape) == p["shape"]
        np.testing.assert_array_equal(got.reshape(-1), np.asarray(t.weights[p["name"]]).reshape(-1))


def test_a_sharded_checkpoint_is_merged_back_into_the_module():
    c = FakeClient()
    m = model()
    t = Trainer(c, "p", m, "cross_entropy", torch.zeros(2, 6))
    names = sorted(t.weights)
    half = len(names) // 2
    parts = {}
    for i, group in enumerate((names[:half], names[half:])):
        trained = {n: np.asarray(t.weights[n]) + 1.0 for n in group}
        parts[f"checkpoint.{i}"] = c.upload_artifact("p", tensors.encode_bundle(trained), "ggb1-f32")
    c.runs["r1"] = {"checkpoint": {"weights": parts}}
    before = {k: v.clone() for k, v in m.state_dict().items()}
    t.load_checkpoint_into(m, "r1")
    for k, v in m.state_dict().items():
        assert torch.allclose(v, before[k] + 1.0), k
    # A part that is not a checkpoint shard is rejected.
    c.runs["r2"] = {"checkpoint": {"weights": {"w1": parts["checkpoint.0"]}}}
    with pytest.raises(ValueError):
        t.checkpoint("r2")


def test_the_grid_chooses_the_levers_or_refuses_before_any_upload():
    x = np.zeros((32, 6), dtype=np.float32)
    y = np.zeros(32, dtype=np.float32)
    c = FakeClient()
    c.plan_answer = {"recommendation": {"segments": 4, "offload_parameters": True,
                                        "task_device_bytes": 1, "fitting_workers": 2},
                     "workers_seen": 3, "advice": "fits"}
    t = Trainer(c, "p", model(), "cross_entropy", torch.from_numpy(x[:2]))
    t.fit(x, y, rounds=1, batch_per_round=8, tasks_per_round=2)
    assert c.created["segments"] == 4 and c.created["offload_parameters"] is True
    # Explicit levers take precedence.
    t.fit(x, y, rounds=1, batch_per_round=8, tasks_per_round=2, segments=2)
    assert c.created["segments"] == 2 and c.created["offload_parameters"] is False
    # Gradient precision: left to the grid (bf16) unless asked; an unknown one fails before upload.
    assert c.created["gradient_precision"] is None
    t.fit(x, y, rounds=1, batch_per_round=8, tasks_per_round=2, segments=2, gradient_precision="f32")
    assert c.created["gradient_precision"] == "f32"
    with pytest.raises(ValueError):
        t.fit(x, y, rounds=1, batch_per_round=8, tasks_per_round=2, gradient_precision="fp16")

    c = FakeClient()
    c.plan_answer = {"recommendation": None, "workers_seen": 2,
                     "advice": "no lever makes a task fit"}
    t = Trainer(c, "p", model(), "cross_entropy", torch.from_numpy(x[:2]))
    import gpugoblin
    with pytest.raises(gpugoblin.PlanError, match="no lever"):
        t.fit(x, y, rounds=1, batch_per_round=8, tasks_per_round=2)
    assert c.store == {}, "nothing may be uploaded before the plan answer"


def test_a_large_dataset_goes_in_row_parts(monkeypatch):
    monkeypatch.setattr(driver, "SINGLE_UPLOAD_BYTES", 6 * 4 * 10)
    x = np.arange(32 * 6, dtype=np.float32).reshape(32, 6)
    y = np.arange(32, dtype=np.float32)
    c = FakeClient()
    t = Trainer(c, "p", model(), "cross_entropy", torch.from_numpy(x[:2]))
    with pytest.warns(UserWarning):
        t.fit(x, y, rounds=1, batch_per_round=8, tasks_per_round=2)
    parts = c.created["dataset_parts"]
    assert set(parts) == {"x"}, parts  # y (128 bytes) fits in one upload
    rows = np.concatenate([tensors.decode_tensor(c.store[r["hash"]]) for r in parts["x"]])
    np.testing.assert_array_equal(rows, x)
    assert len(parts["x"]) == 4


def test_tiles_cut_by_the_grid_are_merged_back_into_the_module():
    """D3: the grid tiled a matrix; the driver reassembles the tiles and the module gets its trained
    weights back unchanged."""
    c = FakeClient()
    m = model()
    t = Trainer(c, "p", m, "cross_entropy", torch.zeros(2, 6))
    # The first Linear, [16, 6] in PyTorch layout, tiled into 3 row tiles.
    name = next(n for n in t.weights if t.weights[n].shape == (16, 6))
    full = np.asarray(t.weights[name]) + 2.0
    tiles = {f"{name}.tile{i}of3": part for i, part in enumerate(np.array_split(full, 3, axis=0))}
    rest = {n: np.asarray(w) + 2.0 for n, w in t.weights.items() if n != name}
    ref = c.upload_artifact("p", tensors.encode_bundle({**tiles, **rest}), "ggb1-f32")
    c.runs["r1"] = {
        "checkpoint": {"weights": {"checkpoint.0": ref}},
        "config": {"graph": {"tiling": [{"param": name, "tiles": 3, "axis": 0}]}},
    }
    got = t.checkpoint("r1")
    np.testing.assert_array_equal(got[name], full)
    assert not any(".tile" in k for k in got)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    t.load_checkpoint_into(m, "r1")
    for k, v in m.state_dict().items():
        assert torch.allclose(v, before[k] + 2.0), k
