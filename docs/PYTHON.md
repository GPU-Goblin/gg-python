# The PyTorch driver and the Python client

The researcher writes ordinary PyTorch. The driver **captures** the model, translates it into a
`gg-graph/v1` description, has it checked against the published operator profile, then launches
collective training on the grid and returns the trained weights. Nothing more to learn.

One line does not move: **no Python code crosses the network.** What leaves is a description and
tensors. Two guarantees depend on it: the promise made to contributors that nothing from a third
party runs on their machine, and verification by recomputation (ADR-0016).

## Installation

```bash
pip install -e .                       # client only, from a checkout of this repository
pip install -e '.[torch]'              # with the PyTorch driver
```

## End to end

```python
import numpy as np, torch
from torch import nn
import gpugoblin as gg

client = gg.Client("http://server:8080", token="ggt_…")
model = nn.Sequential(nn.Linear(6, 24), nn.GELU(), nn.Linear(24, 3))

trainer = gg.Trainer(client, project="…", model=model, loss="cross_entropy",
                     example_input=torch.randn(2, 6), name="my-model")
print(trainer.parameter_count, "parameters")
print(trainer.description_json())        # the description that will be sent, readable

run = trainer.fit(x, y, rounds=25, batch_per_round=128, tasks_per_round=4, lr=0.05)
view = trainer.wait(run["run_id"])       # blocks until the end
trainer.load_checkpoint_into(model, run["run_id"])   # the trained weights come back into the module
```

Measured on 2026-09-17 on a real server with one worker and one validator: three-class
classification, 384 examples, 25 rounds × 4 tasks, **loss from 1.098 to 0.025**, every result
accepted by the validator (`tests/test_live.py`).

**Intermediate checkpoints.** The grid keeps a run's checkpoint at step 0, every 50 steps and at
the current step; the others are removed about ten minutes after their round is settled (ADR-0040), and
downloading one then says it expired. `checkpoint()` always reads the current one.

**Gradient precision.** Each task sends its gradient in bf16 by default: half the bytes of f32, for
a rounding (0.16 % of the norm) that training does not notice (ADR-0035). Weights, the server's
average and the optimizer state stay f32. `fit(..., gradient_precision="f32")` keeps every bit, at
twice the upload per task.

## Through the laboratory's node

A laboratory can install a node (`gg-node`, [deploy/node/INSTALL.md](https://github.com/GPU-Goblin/gg-server/blob/main/deploy/node/INSTALL.md)). The
node keeps the data and the trained weights on the laboratory's machine, and does the optimizer
step there. The control plane then only schedules. To use the node, give its address to the
client; nothing else changes:

```python
client = gg.Client("https://grid.gpugoblin.com", token, node="http://127.0.0.1:7450")
trainer = gg.Trainer(client, project, model, "cross_entropy", x[:2])
run = trainer.fit(x, y, rounds=25, batch_per_round=128, tasks_per_round=4)
trainer.wait(run["run_id"])
trainer.load_checkpoint_into(model, run["run_id"])   # the weights come from the node
```

With a node:

- **Datasets and starting weights** go to the node. The control plane receives only the model's
  description.
- **The run** is created by the node, which draws or checks the starting weights and applies every
  optimizer step.
- **`client.node_run(run_id)`** gives the tasks of the round in progress by state.
  **`client.node_status()`** gives what the node holds and pulls.
- **The node checks your token** with the control plane and acts only for you. It works only for the
  projects it is attached to (`gg node attach`).

Datasets stay on the node whatever their size (up to 8 GiB each); beyond 256 MiB, each task
receives only its micro-batch rows, which the node cuts. A model no worker holds whole is cut into
stages by the node (from the `stage_memory_bytes` the plan recommends), and a matrix larger than a
worker is tiled by the node (`tile_max_param_bytes`); the checkpoint comes back with the tiles
reassembled, as without a node.

## Examples

| Example | Needs | What it shows |
|---|---|---|
| [check_offline.py](../examples/check_offline.py) | `gg` only | capture a module, have it checked as the server would, see an unsupported operator rejected by name |
| [train_classifier.py](../examples/train_classifier.py) | a grid | train on verified workers and load the weights back into the module |
| [plan_before_training.py](../examples/plan_before_training.py) | a grid | what the grid proposes (levers, stages, tiles) before any data is sent |

The grid examples read `GG_SERVER`, `GG_TOKEN` and `GG_PROJECT`; the project must admit
`graph.grad.v1` (and `graph.stage.v1` for staged models).

## What capture accepts

`torch.export` yields the whole graph in ATen; the driver translates it operator by operator into
the `gg-ops/8` profile:

| PyTorch construct | Profile operator |
|---|---|
| `nn.Linear`, `@`, `torch.matmul`, `addmm`, `bmm` | MatMul (+ Transpose, Add) |
| `nn.ReLU`, `nn.GELU`, `nn.Sigmoid`, `nn.Tanh` | Relu, Gelu, Sigmoid, Tanh |
| `softmax` | Softmax |
| `nn.LayerNorm`, `nn.RMSNorm` | LayerNorm, RMSNorm |
| `nn.Conv2d`, `nn.MaxPool2d`, `nn.AvgPool2d`, `nn.BatchNorm2d` | Conv2d, MaxPool2d, AvgPool2d, BatchNorm2d |
| `view`, `reshape`, `flatten`, `transpose`, `permute`, `t()`, `cat` | Reshape, Flatten, Transpose, Concat |
| `nn.Embedding` | Gather |
| `nn.Dropout` | Dropout (drawn from the protocol's deterministic generator) |
| `+ - * /` between tensors, or with a scalar | Add, Sub, Mul, Div; the scalar becomes a frozen constant |
| causal attention: triangular buffer and `masked_fill(-inf)`, `torch.triu` in `forward`, `torch.where` | CausalMask |
| `F.scaled_dot_product_attention(q, k, v, is_causal=…)` | Transpose, MatMul, Mul, CausalMask, Softmax, MatMul |
| `x[..., a:b]`, contiguous slices | Slice |
| learned positions `wpe(torch.arange(t))` | the parameter itself, or its slice through Slice |
| sinusoidal table in a buffer, RoPE `cos` and `sin` tables, scaled or not | frozen parameters with a named generator, recognized by their value |
| `chunk`, `split`, `unbind`, `select`, `squeeze`, `unsqueeze`, `stack` | Slice, Reshape, Concat |
| `nn.Conv1d` | Conv2d of height 1, between two Reshape |
| `nn.GroupNorm` | ReduceMean, Sub, Mul, Formula (`rsqrt(var + ε)`) |
| `nn.AdaptiveAvgPool2d`, `x.mean((2, 3))` | ReduceMean, or AvgPool2d when the output divides the input |
| `nn.LSTM` (batch first, unidirectional, stacked layers) | unrolled in time: MatMul, Slice, Formula, Concat |
| `exp`, `log`, `sqrt`, `abs`, `sin`, `cos`, `erf`, `pow`, `softplus`, `clamp`, `leaky_relu`, `elu`, `mish`, `hardswish`, `maximum`, … | Formula |
| `gpugoblin.formula(expr, …)` | Formula, as written |
| `expand` over the batch (ViT class token) | Formula |
| empty attention mask, or additive mask computed in the graph | none, or Add on the scores |
| `nn.ConvTranspose1d`, `nn.ConvTranspose2d` (HiFi-GAN vocoders) | ConvTranspose2d (1D: height 1, between two Reshape) |
| `weight_norm` (`_weight_norm`, parametrized or not: Wav2Vec2, vocoders) | Mul, ReduceSum, Formula; `g` and `v` stay the trained parameters |
| `F.pad` constant (negative: a crop), reflect, replicate, circular | Pad (constant, reflect, edge), Slice + Concat for circular |
| `F.interpolate` nearest, linear, bilinear (`align_corners` or not) | Resize |
| `flip`, `roll`, `repeat` | Flip, Slice + Concat, Concat |
| `sum`, `amax`, `amin`, `max(dim)` / `min(dim)` values, `prod`, `cumsum` | ReduceSum, ReduceMax, ReduceMin, ReduceProd, CumSum |
| `var`, `std`, `var_mean`, `logsumexp`, `log_softmax`, `linalg.vector_norm`, `F.normalize` | ReduceMean, ReduceMax, ReduceSum, Formula |
| `index_select`, `x[:, idx]`, `index_add`, `gather` / `scatter_add` along a vector, with a **constant** index (buffer); an embedding looked up with a constant index (T5's relative position buckets) | IndexSelect, IndexAdd |
| `argmax`, `argmin`, `max(dim)` / `min(dim)` with their indices | ArgMax, ArgMin (+ GatherElements for the values) |
| `topk`, `sort`, `argsort` (values and indices) | ArgTopK, ArgSort + GatherElements |
| `gather`, `scatter`, `scatter_add`, `scatter_reduce` (sum, mean) at a **computed** index; `table[idx]` of a parameter | GatherElements, ScatterElements, Gather |
| `.float()` of an index, `repeat_interleave`, `expand` of an index, a `clamp` that changes nothing | Cast, Concat, Reshape |
| a mixture of experts of `transformers` after `gpugoblin.batched_experts(model)` | ArgTopK, Gather of the experts' weights, MatMul |
| `tan`, `asin` … `atanh`, `atan2`, `floor`, `ceil`, `round` (decimals too), `trunc`, `sign`, `remainder`, `fmod`, `log2`, `log10`, `lerp`, `addcmul`, `selu`, `celu`, `logsigmoid`, `hypot`, rounding divisions, `clamp` by tensors, … | Formula |

**Tied weights are one parameter.** A language-model head that is the embedding matrix
(`tie_word_embeddings`: GPT-2, Gemma, T5, Mamba, SmolLM, BERT's decoder…) is lifted by
`torch.export` once per alias; the driver recognizes tensors that share storage and makes them
**one** parameter of the description, named after the first alias in `named_parameters` order and
read by every node that uses it. Its gradient sums both uses, as in PyTorch — on one machine and
across stages, where each stage that uses it contributes a partial gradient that the grid adds. A
checkpoint reloads into that alias, and therefore into all of them. Such a matrix cannot be tiled
(`graph::tiling` refuses it by name). Before 2026-09-24, the aliases were two parameters trained
apart: a description captured before then has to be captured again.

**A description holds at any number of examples.** `torch.export` writes shapes as numbers: a
reshape to `[tokens·k, …]` in a mixture of experts carries the capture's count of tokens. The driver
therefore captures the module twice — at the example batch and at one more example — and compares
the two descriptions: a dimension that follows the number of examples becomes the free leading
dimension (`-1`) of its `Reshape`; anything else that differs is refused by name. Capture with a
batch of at least 2.

Everything else **makes capture fail, naming the operator**, before anything is sent:

```
UnsupportedOperator: operator 'aten.one_hot.default' (node 'one_hot') is not part of profile
gg-ops/8; an output whose shape depends on the data, or an assignment by index, is outside the
profile (ADR-0029). Profile operators: Add, ArgMax, ArgMin, ArgSort, ArgTopK, AvgPool2d, …
```

This is the direct answer to the risk "if some operations are not handled, it will not work": the
researcher finds out in one second and knows what to ask for. `gg.coverage(module, example)` lists
in a single pass **everything** that is missing, with the reason, instead of stopping at the first.

## A custom operator: `gpugoblin.formula`

An element-wise function that exists nowhere else is written as an expression, inside the module:

```python
import gpugoblin as gg

class Block(nn.Module):
    def forward(self, x):
        h = self.lin(x)
        return gg.formula("a * tanh(softplus(a)) + 0.1 * b", h, self.gain)   # a = h, b = gain
```

In PyTorch, `formula` computes the expression with torch, gradient included: the module trains and
evaluates locally as usual. At capture time, it becomes **one** `Formula` node in the description,
which the Rust executor evaluates with the same grammar ([gg-core/docs/MODELS.md](https://github.com/GPU-Goblin/gg-core/blob/main/docs/MODELS.md)). The expression is checked
on the first call, with the validator's messages:

```
FormulaError: expression: 'pow' is neither an input (a, b, c, d) nor a known function (exp, log, …)
```

No code is sent: only the text of the expression, whose grammar is closed (four operations, power
with a literal exponent, thirty functions — among them, since `gg-ops/7`, the inverse
trigonometric and hyperbolic functions, `atan2` and the rounding functions, whose derivative is
zero) and bounded. For a block richer than an expression, an
ordinary `nn.Module` is enough: the driver captures its composition.

## Which architectures pass: the corpus

`python tools/coverage_corpus.py` captures 26 miniature architectures and, for each one, has the
Rust tool validate and execute the description, then compares it with PyTorch
(`tests/test_corpus.py`). Each description is executed at the capture batch **and at two other
batch sizes** (3 and 5), and compared with PyTorch at each: a shape that kept the capture's number
of examples fails there, not on the grid. As of 2026-09-24 (ADR-0029), all of them compute the same
function as PyTorch:

| Architecture | Nodes | Relative difference from PyTorch |
|---|---|---|
| Llama, Qwen2, Mistral, Phi-3 | 135 to 141 | ≤ 4.2e-7 |
| Gemma (tanh GELU, computed as such since `gg-ops/8`) | 139 | 3.1e-7 |
| GPT-2, GPT-NeoX | 101, 109 | ≤ 3.6e-7 |
| BERT, DistilBERT | 92, 87 | ≤ 3.6e-7 |
| T5 (relative position bias: IndexSelect) | 265 | 3.9e-7 |
| ViT, ResNet, ConvNeXt | 92, 32, 40 | ≤ 4.8e-7 |
| Whisper (encoder) | 53 | 1.8e-7 |
| Mamba | 321 | 2.4e-7 |
| Mixtral, Qwen2-MoE, OLMoE, Qwen3-MoE (top-2 of 4 experts, `batched_experts`) | 179 to 209 | ≤ 3.0e-7 |
| SpeechT5 HiFi-GAN, FastSpeech2 HiFi-GAN (vocoders: ConvTranspose2d) | 70, 70 | ≤ 1.9e-8 |
| Wav2Vec2 (weight-normalized positional convolution) | 120 | 8.8e-7 |
| LSTM, Conv1d + GroupNorm | 96, 18 | ≤ 1.6e-7 |
| GNN over a fixed graph (`index_add`: IndexAdd) | 7 | 8.3e-8 |
| Mixture of experts, hand-written top-2 gating (`topk`, `scatter`) | 32 | 2.5e-7 |

Before `gg-ops/7`, 16 of the first 18 passed; before `gg-ops/8`, the mixtures of experts were
rejected (`topk`). A Mixtral and the hand-written mixture also **train** as in PyTorch: loss and
gradient of every parameter, the router's included, within 4.4e-7
(`crates/gg-compute/tests/graph_ops8.rs`).

### Mixtures of experts

By default, `transformers` runs each expert on the tokens routed to it: a loop whose shapes depend
on the routing, which no static graph expresses. `gpugoblin.batched_experts(model)` switches the
model to `transformers`' own `batched_mm` implementation — the same function, with static shapes:
each token is sent to its `k` experts by gathering their weights.

```python
model = transformers.MixtralForCausalLM.from_pretrained(...)
gg.batched_experts(model)          # before capture
description, weights = gg.export_model(model, ids, "cross_entropy", input_dtypes=["i64"])
```

Without it, capture names the operator and says what to call. The cost: each token materializes
the weights of its `k` experts (`[tokens·k, out, in]`), as `batched_mm` does in PyTorch.

## Causal attention

The four common ways of writing causal attention are captured, and for each one the test checks
that **the Rust executor computes the same function as PyTorch**, within 1e-5
(`tests/test_causal.py`):

```python
# nanoGPT: mask in a buffer, explicit 1/√d factor
self.register_buffer("bias", torch.tril(torch.ones(T, T)).view(1, 1, T, T))
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

# the modern way
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

The driver does not guess from how the code is written: it computes the value of the mask and only
translates it into `CausalMask` if it is exactly the triangle of future positions. A mask that
comes close without being exactly that is rejected, with the reason.

Measured on 2026-09-21 on a real server with a worker and a validator: a language model written
like nanoGPT learns to predict the next token, **loss from 2.99 to 0.0000, accuracy 100 %** in 40
rounds.

## Positional encoding

nanoGPT-style learned positions, a sinusoidal table in a buffer, and Hugging Face-style RoPE are
captured without changing anything in the model code:

```python
# RoPE: inv_freq in a buffer, cos and sin computed in forward, rotate_half half-rotation
freqs = torch.outer(torch.arange(t, dtype=torch.float32), self.inv_freq)
emb = torch.cat((freqs, freqs), dim=-1)
q = q * emb.cos()[None, None] + rotate_half(q) * emb.sin()[None, None]
```

The driver recognizes the tables **by their value**, estimates their base (10,000, 500,000…) and
describes them by their generator; the checkpoint carries the module's exact values. For each way
of writing it, the Rust executor's output is within 5e-7 of PyTorch
(`tests/test_positions.py`).

Measured on 2026-09-21 on a real server: a RoPE model written this way learns to copy the token two
positions earlier, **accuracy from 6 % to 92 %** in 60 rounds.

### Scaled RoPE: Llama 3.1, YaRN, LongRoPE

The rotary classes of `transformers` pass as is, scaling included:

```python
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

self.rope = LlamaRotaryEmbedding(config)       # config.rope_parameters: llama3, yarn, linear…
cos, sin = self.rope(x, torch.arange(t)[None])
q, k = apply_rotary_pos_emb(q, k, cos, sin)
```

The driver does not read the configuration: it recovers the **frequencies** and the **amplitude**
from the value of the pair of tables (cosine, sine), whatever variant produced them, and writes
them into the description (`{"rotary_cos": {"frequencies": [...], "amplitude": 1.1386}}`). Across
sixteen `transformers` configurations, the recovered frequencies are the module's exact f32
frequencies. The original law remains described by its base. A partial rotation (Phi, GPT-NeoX),
where only part of the head rotates, also passes (`tests/test_rope_scaling.py`).

**Dynamic NTK and LongRoPE** choose their frequencies according to the length, in the forward
pass. `torch.export` rejects that choice, and the driver says so while pointing to the solution:
freeze the module at the training length first.

```python
from gpugoblin import freeze_rotary_embeddings

freeze_rotary_embeddings(model, seq_len)   # runs each rotary module once, then freezes it
description, weights = export_model(model, ids[:2], "cross_entropy", input_dtypes=["i64"])
```

The model's output at that length does not change; its frequencies no longer adapt to another
length, which capture does not accept anyway.

Measured on 2026-09-21 on a real server: the `transformers` rotary class with YaRN (amplitude
1.139) learns the same task, **accuracy from 6.0 % to 93.3 %** in 60 rounds; replayed, same
figures.

## `transformers` models, as is

Since `gg-ops/5` ([gg-server/docs/adr/0023-hugging-face-models.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/adr/0023-hugging-face-models.md)), a `LlamaForCausalLM`, a
`Qwen2ForCausalLM` or a `MistralForCausalLM` passes without a single adapted line: hand-written
RMSNorm, SwiGLU network, grouped-query attention (GQA), scaled RoPE. All you need is to return the
logits in the shape the loss expects:

```python
from transformers import LlamaForCausalLM

class Logits(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
    def forward(self, ids):
        return self.lm(input_ids=ids, use_cache=False).logits.reshape(-1, vocab)

trainer = gg.Trainer(client, project, Logits(LlamaForCausalLM(config)), "cross_entropy", ids[:2],
                     input_dtypes=["i64"], target_shape=[0, seq])
```

For Llama (4, 2 and 1 key heads), Qwen2, Mistral and scaled Llama 3.1, the Rust executor computes
the same function as PyTorch within 1.5e-7 (`tests/test_hf_models.py`). A frozen base
(`requires_grad=False`) stays frozen: only the trained parameters receive a gradient and travel.

## Before submitting: `trainer.plan()`

```python
plan = trainer.plan(batch_per_round=128, tasks_per_round=4)
print(plan["advice"])   # "the task fits with 4 activation recomputation segment(s)"
```

The grid answers based on what its workers advertise (memory, profile): required profile, memory of
a task without any lever, number of workers that can carry it, and the cheapest memory levers that
make it fit. `fit()` uses this on its own when neither `segments` nor `offload_parameters` is given,
and **refuses before any data is sent** if no worker can carry the task, saying what to do (reduce
the micro-batch, or split the model into stages).

**A model larger than every one of the machines.** If no worker can carry a task for the whole
model, even with the memory levers, the plan proposes to split it into **stages** spread across
workers ([gg-server/docs/adr/0024-staged-execution.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/adr/0024-staged-execution.md)); if a single matrix exceeds a machine, it first
proposes to **tile** it. `fit()` applies the proposal, and `checkpoint()` stitches the tiles back
together: the module receives its weights as if it had been trained whole. Measured on the real
grid: a Llama on three workers that each have only a third of the required memory, 4 stages,
accuracy from 5.4 % to 92.2 %. And a 4 MiB matrix on 3 MB workers: `fit()` tiles it across 24
stages, and `load_checkpoint_into` gives the module back its whole matrices
(`test_a_matrix_larger_than_every_worker_is_split_on_the_grid`). The price: about ten times slower
per round on a small model ([gg-server/docs/STATUS.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/STATUS.md), P11 D6).

Data and weights are sent at the right size: beyond 256 MiB, a dataset is sent in row shards and
each task only receives the rows of its micro-batch; initial weights are sent parameter by
parameter, and the checkpoint is split into shards ([gg-server/docs/ROADMAP.md](https://github.com/GPU-Goblin/gg-server/blob/main/docs/ROADMAP.md), P11).

## What capture does not do

- **No data-dependent control flow.** `torch.export` rejects a loop whose number of iterations
  depends on a tensor. The graph is static; that is what makes it verifiable.
- **No dynamic dimensions other than the batch.** The sequence length is fixed at capture time.
- **No interleaved RoPE** (GPT-J: stride-two slices), **no multimodal RoPE** (Qwen2-VL), **no
  ALiBi**: their tables or slices are not translated, and capture says so. Rotary modules with
  several layer types (Gemma 3) cannot be frozen with `freeze_rotary_embeddings`.
- **Attention masks: causal, empty or computed.** A constant mask is recognized **by its value**:
  the strict triangle of future positions (boolean, or additive with "−∞"), or a mask that masks
  nothing. An additive mask computed in the graph (from a padding mask supplied as an input) is
  added to the scores. A constant sliding window, a computed boolean mask or dropout inside
  attention are rejected, with a message saying so.
- **No shapes that depend on the data** (`nonzero`, `torch.where` with one argument, boolean
  masking), no assignment by index (`index_put`), no `one_hot`, no **integer arithmetic**: integers
  are indices, computed by `argmax`, `topk`, `sort` and read by gathers and scatters (ADR-0029). A
  `scatter` (assignment) is accepted when its positions are known to be distinct (the indices of a
  `topk` or a `sort` along the same axis).
- **No bidirectional LSTM or LSTM with a supplied initial state** (ADR-0025).
- `remainder` and `fmod` are written `a − b·floor(a/b)` and `a − b·trunc(a/b)`: exact when the
  quotient is, otherwise within one unit in the last place of `a` (ADR-0027).
- **A hand-written derivative** (`torch.autograd.Function`) is not carried over: the forward pass
  is captured, and the derivative is the one from automatic differentiation.
- **No bf16**: capture and execution are in f32.
- **Non-trainable buffers** (BatchNorm running means) are not carried over: the graph trains, it
  does not run inference.

## The binary formats, without PyTorch

`gpugoblin.tensors` reads and writes the protocol's two formats, with NumPy alone:

```python
from gpugoblin import tensors
tensors.write_tensor("train_x.ggt", x)             # GGT1: one f32 tensor
weights = tensors.read_bundle("checkpoint.ggb")    # GGB1: named tensors
```

A `GGB1` sorts its entries by name: the same set of tensors gives the same bytes, and therefore the
same hash, whatever the insertion order. `gg data inspect` reads both, and `tests` checks
that the Rust binary reads back what Python writes.

## The client alone

Without PyTorch, `gg.Client` is enough to drive the grid from a script:

```python
client.operator_profile()          # GET /v1/catalog/ops, the accepted profile
client.upload_artifact(project, data, "ggt1-f32")
client.create_graph_run(project, graph_ref, datasets, rounds=…, batch_per_round=…, tasks_per_round=…)
client.wait_run(run_id)
```

A computation "outside AI" that splits into independent tasks goes through the same calls: they
are descriptions and tensors, not code.

## Troubleshooting

| Message | Cause |
|---|---|
| `UnsupportedOperator: …` | operator outside the profile; `client.operator_profile()` lists what passes |
| `graph refused: …` (HTTP 422) | the description is malformed; `gg job check-model` says where |
| `graph input '…' has no dataset` | the model takes several inputs: pass `extra_inputs` |
| `the checkpoint does not cover: […]` | the module has changed since capture |
