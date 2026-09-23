# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] — unreleased

First public version of the `gpugoblin` package.

- REST client of the control plane; `GGT1`/`GGB1` binary formats in NumPy.
- PyTorch driver: capture with `torch.export`, translation into `gg-graph/v1` against the `gg-ops/6`
  operator profile with named rejections, `transformers` models as they are, rotary and learned
  positional encodings, causal masks, custom element-wise operators (`gg.formula`).
- `Trainer`: plan, upload, collective training, sharded checkpoints, staged and tiled models
  reassembled on return, weights loaded back into the module.
- Profile `gg-ops/7`: transposed convolutions, weight norm, padding, upsampling, reductions,
  var/std/logsumexp/log_softmax/norms, flip/roll/repeat, constant-index indexing and about thirty
  element-wise functions; T5, a GNN, HiFi-GAN vocoders and Wav2Vec2 capture and match PyTorch.
- Profile `gg-ops/8`: argmax/argmin, topk, sort/argsort, gather and scatter at computed indices;
  `gpugoblin.batched_experts` for `transformers` mixtures of experts (Mixtral, Qwen2-MoE, OLMoE,
  Qwen3-MoE).
- Descriptions no longer depend on the capture batch: the driver captures twice and derives the
  batch dimension; every corpus model is checked at other batch sizes.
- Tied weights (a language-model head sharing the embedding) are one parameter, as in PyTorch.
- Capturing a module gives it back in the mode it came in (it was left in evaluation mode).
- The client waits and replays on 429 and 503, honouring `Retry-After`, like the Rust client.
- Examples: offline check, training on a grid, planning before training.

The full history of the project, before it was split into several repositories, is summarized in
[gg-server/CHANGELOG.md](https://github.com/GPU-Goblin/gg-server/blob/main/CHANGELOG.md).
