"""Capture a PyTorch model and have it checked, without a server.

The driver translates the module into a `gg-graph/v1` description; the `gg` command line then
validates it exactly as the server would, and says how much memory a task needs. An operator
outside the published profile is rejected by name, before anything is sent anywhere.

    GG_BIN=/path/to/gg python examples/check_offline.py

`gg` is built from the gg-server repository (`cargo build -p gg-cli`); without `GG_BIN`, it is
looked up on `PATH`.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch
from torch import nn

import gpugoblin as gg


def gg_binary() -> str:
    found = os.environ.get("GG_BIN") or shutil.which("gg")
    if not found:
        sys.exit("gg not found: set GG_BIN or put gg on PATH (cargo build -p gg-cli in gg-server)")
    return found


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.LayerNorm(64), nn.Linear(64, 4))

    def forward(self, x):
        return self.body(x)


def main():
    torch.manual_seed(0)
    model = Classifier()
    description, weights = gg.export_model(model, torch.randn(2, 16), "cross_entropy", name="classifier")
    operators = sorted({node["op"] for node in description["nodes"]})
    print(f"captured {len(description['nodes'])} nodes, {len(weights)} parameter tensors: {operators}")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "classifier.json")
        with open(path, "w") as f:
            json.dump(description, f)
        out = subprocess.run([gg_binary(), "job", "check-model", path, "--batch", "128"],
                             capture_output=True, text=True)
        print(out.stdout.strip() or out.stderr.strip())
        if out.returncode != 0:
            sys.exit(out.returncode)

    # An operator outside the profile is named, with the PyTorch node that uses it.
    class OneHot(nn.Module):
        def forward(self, x):
            return nn.functional.one_hot(x.argmax(-1), 8).float()

    try:
        gg.export_model(OneHot(), torch.randn(2, 8), "mse")
    except gg.UnsupportedOperator as e:
        print(f"rejected as expected: {str(e).split('. Profile operators')[0]}")
    else:
        sys.exit("one_hot should have been rejected")


if __name__ == "__main__":
    main()
