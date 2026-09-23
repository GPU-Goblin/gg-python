# gpugoblin

The Python package and PyTorch driver of [GPU Goblin](https://github.com/GPU-Goblin/gg-server): write
ordinary PyTorch, train it on volunteer GPUs, get verified weights back.

```python
import gpugoblin as gg
from torch import nn

client = gg.Client("http://server:8080", token="ggt_…")
model = nn.Sequential(nn.Linear(6, 24), nn.GELU(), nn.Linear(24, 3))

trainer = gg.Trainer(client, project="…", model=model, loss="cross_entropy", example_input=x[:2])
run = trainer.fit(x, y, rounds=25, batch_per_round=128, tasks_per_round=4, lr=0.05)
trainer.wait(run["run_id"])
trainer.load_checkpoint_into(model, run["run_id"])   # the trained weights come back
```

The model is **captured** with `torch.export`, translated into a `gg-graph/v1` description and
checked against the published operator profile: an operator outside the profile is rejected, by
name, before anything is sent. **No Python code ever goes over the network**: what leaves is a
description and tensors, computed by GPU Goblin's own code on the workers and recomputed by a
validator before any result is accepted.

What it handles today: MLPs, CNNs (ResNet, ConvNeXt), ViT, recurrent networks unrolled, causal
transformers written like nanoGPT, `transformers` models as they are (Llama, Qwen2, Mistral, Phi-3,
Gemma, GPT-2, BERT, T5, Mamba…), mixtures of experts (Mixtral, Qwen2-MoE, OLMoE, Qwen3-MoE, after
`gg.batched_experts(model)`), and speech models (Whisper's encoder, Wav2Vec2, HiFi-GAN vocoders).
What is refused is refused by name, before anything is sent: shapes that depend on the data
(boolean masking, `nonzero`, text-to-speech models such as VITS). A model larger than
any single worker is split into stages, and a matrix larger than any worker is tiled, without any
change to your code: `trainer.plan()` says what the grid can do before any data is sent.

> **Status: research software, version 0.1.0.** There is no public GPU Goblin network yet: you need
> access to a grid (a server, workers and a validator; see
> [gg-server](https://github.com/GPU-Goblin/gg-server) to run one locally).

## Installation

```bash
git clone https://github.com/GPU-Goblin/gg-python
cd gg-python
pip install -e .              # client and binary formats only (numpy, requests)
pip install -e '.[torch]'     # with the PyTorch driver
```

## Documentation and examples

- [docs/PYTHON.md](docs/PYTHON.md) — the driver end to end: what capture accepts, custom
  element-wise operators (`gg.formula`), causal attention, positional encodings, `transformers`
  models, planning, troubleshooting.
- [examples/check_offline.py](examples/check_offline.py) — capture a model and have it checked
  without a server.
- [examples/train_classifier.py](examples/train_classifier.py) — train on a grid and load the
  weights back.
- [examples/plan_before_training.py](examples/plan_before_training.py) — ask the grid what it can do
  with a model.
- The model description format and the operator profile:
  [gg-core/docs/MODELS.md](https://github.com/GPU-Goblin/gg-core/blob/main/docs/MODELS.md).

## Tests

```bash
pip install -e '.[torch,test]'
GG_BIN=/path/to/gg python -m pytest -q tests
```

Many tests use the `gg` command line as the reference checker: it is built from
[gg-server](https://github.com/GPU-Goblin/gg-server) with `cargo build -p gg-cli`. `GG_BIN` names it;
otherwise `gg` is looked up on `PATH`, and the tests that need it are skipped, saying so.

`tests/test_live.py` trains real models on a real grid. It runs when `GG_SERVER`, `GG_TOKEN` and
`GG_PROJECT` are set; `GG_STAGED_DEVICE` and `GG_TILED_DEVICE` enable the staged and tiled cases.

## License

[PolyForm Noncommercial 1.0.0](LICENSE): free for research and any noncommercial use,
source-available, not open source in the OSI sense. See [LICENSING.md](LICENSING.md).
