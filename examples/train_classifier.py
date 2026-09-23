"""Train an ordinary PyTorch classifier on a GPU Goblin grid and get the weights back.

    GG_SERVER=http://127.0.0.1:8080 GG_TOKEN=ggt_... GG_PROJECT=<project id> \
        python examples/train_classifier.py

The project must admit `graph.grad.v1` and have credits. Every round is split into micro-batch
tasks, computed by workers, recomputed by a validator, and aggregated by the server into one
optimizer step; the trained weights are then loaded back into the module.
"""

import os

import numpy as np
import torch
from torch import nn

import gpugoblin as gg


def main():
    client = gg.Client(os.environ["GG_SERVER"], os.environ["GG_TOKEN"])
    project = os.environ["GG_PROJECT"]

    # Three classes, told apart by the first coordinate.
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, size=384)
    x = rng.standard_normal((384, 6)).astype(np.float32) * 0.3
    x[:, 0] += y * 2.0

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 24), nn.GELU(), nn.Linear(24, 3))

    def loss_of(m):
        with torch.no_grad():
            return float(nn.functional.cross_entropy(m(torch.from_numpy(x)), torch.from_numpy(y)))

    before = loss_of(model)
    trainer = gg.Trainer(client, project, model, "cross_entropy", torch.from_numpy(x[:2]), name="classifier")
    print(f"{trainer.parameter_count} parameters; submitting 25 rounds of 128 examples in 4 tasks")
    run = trainer.fit(x, y.astype(np.float32), rounds=25, batch_per_round=128, tasks_per_round=4, lr=0.05)
    view = trainer.wait(run["run_id"], timeout=600)
    if view["state"] != "completed":
        raise SystemExit(f"run {run['run_id']} ended in state {view['state']}: {view.get('failure')}")

    trainer.load_checkpoint_into(model, run["run_id"])
    print(f"loss {before:.4f} -> {loss_of(model):.4f} after {view['step']} verified rounds")


if __name__ == "__main__":
    main()
