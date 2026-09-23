"""Ask the grid what it can do with a model before sending any data.

    GG_SERVER=http://127.0.0.1:8080 GG_TOKEN=ggt_... GG_PROJECT=<project id> \
        python examples/plan_before_training.py

The plan compares the memory one task needs with what the connected workers announce, and
proposes memory levers (activation recomputation, parameter offloading), a split into stages, or
tiling a matrix too large for any single machine. `Trainer.fit()` applies the proposal by itself;
this example only prints it.
"""

import os

import torch
from torch import nn

import gpugoblin as gg


def main():
    client = gg.Client(os.environ["GG_SERVER"], os.environ["GG_TOKEN"])
    project = os.environ["GG_PROJECT"]
    capacity = client.grid_capacity()
    print(f"grid: {json_line(capacity)}")

    torch.manual_seed(0)
    model = nn.Sequential(*[nn.Sequential(nn.Linear(256, 256), nn.GELU()) for _ in range(8)], nn.Linear(256, 10))
    trainer = gg.Trainer(client, project, model, "cross_entropy", torch.randn(2, 256), name="deep-mlp")
    plan = trainer.plan(batch_per_round=512, tasks_per_round=4)
    print(f"{trainer.parameter_count} parameters, {trainer.checkpoint_bytes} bytes of weights")
    print(f"advice: {plan.get('advice')}")
    for key in ("recommendation", "staged", "tiling"):
        if plan.get(key):
            print(f"{key}: {json_line(plan[key])}")


def json_line(value) -> str:
    import json

    return json.dumps(value, sort_keys=True)


if __name__ == "__main__":
    main()
