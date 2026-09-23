"""Coverage of the operator profile over a corpus of real models (gg-core/docs/MODELS.md).

Each model is a miniature version (a few layers, width 32) of a `transformers` architecture —
text, vision, and since `gg-ops/7` speech (two HiFi-GAN vocoders, Wav2Vec2) — or of a
hand-written classic. For each one: captured as is, or the list of what is missing. The final
table ranks the missing operators by the number of models they block: that is the order in which
to add them.

A captured model is then **verified** (`verify`): the description is validated by the Rust tool,
executed by it on the example, and its output compared with PyTorch's. Capturing is not enough;
the gap is what says whether the translation is right. `tests/test_corpus.py` turns this into a
test.

    cargo build -p gg-cli                      # the `gg` binary that validates and executes
    python tools/coverage_corpus.py            # Markdown table on standard output
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from gpugoblin import batched_experts, coverage, export_model, tensors

logging.disable(logging.WARNING)
torch.manual_seed(0)
V, T, B = 64, 8, 2


class Logits(nn.Module):
    """A `transformers` model, reduced to one input and one output."""

    def __init__(self, inner, call):
        super().__init__()
        self.inner, self.call = inner, call

    def forward(self, x):
        return self.call(self.inner, x)


def ids():
    return torch.randint(0, V, (B, T))


def text(cls, config, **extra):
    import transformers as tf

    m = getattr(tf, cls)(getattr(tf, config)(**extra)).eval()
    return Logits(m, lambda m, x: m(input_ids=x, use_cache=False).logits), ids(), ["i64"]


class Moe(nn.Module):
    """Mixture of experts with top-2 routing (Mixtral, Qwen-MoE): the experts chosen depend on
    the data, through integer indices."""

    def __init__(self):
        super().__init__()
        self.router = nn.Linear(16, 4)
        self.experts = nn.ModuleList(nn.Linear(16, 16) for _ in range(4))

    def forward(self, x):  # x: [batch, T, 16]
        scores = self.router(x)
        top, idx = scores.topk(2, dim=-1)
        gate = torch.zeros_like(scores).scatter(-1, idx, top.softmax(-1))
        return sum(gate[..., e:e + 1] * expert(x) for e, expert in enumerate(self.experts))


def corpus():
    import transformers as tf

    # Every model is initialized from the same seed, whatever ran before (tests in any order).
    torch.manual_seed(0)

    small = dict(vocab_size=V, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                 intermediate_size=64, max_position_embeddings=32)
    yield "Llama", *text("LlamaForCausalLM", "LlamaConfig", num_key_value_heads=2, **small)
    yield "Qwen2", *text("Qwen2ForCausalLM", "Qwen2Config", num_key_value_heads=2, **small)
    yield "Mistral", *text("MistralForCausalLM", "MistralConfig", num_key_value_heads=2, sliding_window=None, **small)
    yield "Gemma", *text("GemmaForCausalLM", "GemmaConfig", num_key_value_heads=2, head_dim=8, **small)
    yield "Phi-3", *text("Phi3ForCausalLM", "Phi3Config", num_key_value_heads=2, pad_token_id=0, **small)
    yield "GPT-2", *text("GPT2LMHeadModel", "GPT2Config", vocab_size=V, n_embd=32, n_layer=2, n_head=4, n_positions=32)
    yield "GPT-NeoX", *text("GPTNeoXForCausalLM", "GPTNeoXConfig", **small)
    yield "BERT", *text("BertForMaskedLM", "BertConfig", **small)
    yield "DistilBERT", *text("DistilBertForMaskedLM", "DistilBertConfig", vocab_size=V, dim=32, n_layers=2, n_heads=4, hidden_dim=64, max_position_embeddings=32)
    t5 = tf.T5ForConditionalGeneration(tf.T5Config(vocab_size=V, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_heads=4, decoder_start_token_id=0)).eval()
    yield "T5", Logits(t5, lambda m, x: m(input_ids=x, decoder_input_ids=x, use_cache=False).logits), ids(), ["i64"]
    vit = tf.ViTForImageClassification(tf.ViTConfig(image_size=16, patch_size=4, num_channels=3, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64, num_labels=5)).eval()
    yield "ViT", Logits(vit, lambda m, x: m(pixel_values=x).logits), torch.randn(B, 3, 16, 16), None
    resnet = tf.ResNetForImageClassification(tf.ResNetConfig(num_channels=3, embedding_size=16, hidden_sizes=[16, 32], depths=[1, 1], num_labels=5)).eval()
    yield "ResNet", Logits(resnet, lambda m, x: m(pixel_values=x).logits), torch.randn(B, 3, 32, 32), None
    convnext = tf.ConvNextForImageClassification(tf.ConvNextConfig(num_channels=3, num_stages=2, hidden_sizes=[16, 32], depths=[1, 1], num_labels=5)).eval()
    yield "ConvNeXt", Logits(convnext, lambda m, x: m(pixel_values=x).logits), torch.randn(B, 3, 32, 32), None
    whisper = tf.WhisperModel(tf.WhisperConfig(vocab_size=V, num_mel_bins=8, d_model=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=4, decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64, max_source_positions=8, max_target_positions=8, pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1)).eval()
    yield "Whisper (encoder)", Logits(whisper.encoder, lambda m, x: m(input_features=x).last_hidden_state), torch.randn(B, 8, 16), None
    mamba = tf.MambaForCausalLM(tf.MambaConfig(vocab_size=V, hidden_size=32, state_size=4, num_hidden_layers=2, expand=2, conv_kernel=4)).eval()
    yield "Mamba", Logits(mamba, lambda m, x: m(input_ids=x, use_cache=False).logits), ids(), ["i64"]
    # Mixtures of experts (ADR-0029), top-2 routing over 4 experts, with transformers' batched_mm
    # experts (`gpugoblin.batched_experts`): static shapes, the weights of the chosen experts
    # gathered per token.
    moe = dict(small, intermediate_size=32, num_key_value_heads=2)
    for label, cls, config, extra in [
        ("Mixtral", "MixtralForCausalLM", "MixtralConfig", dict(num_local_experts=4, num_experts_per_tok=2)),
        ("Qwen2-MoE", "Qwen2MoeForCausalLM", "Qwen2MoeConfig",
         dict(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16, shared_expert_intermediate_size=32)),
        ("OLMoE", "OlmoeForCausalLM", "OlmoeConfig", dict(num_experts=4, num_experts_per_tok=2)),
        ("Qwen3-MoE", "Qwen3MoeForCausalLM", "Qwen3MoeConfig",
         dict(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16, head_dim=8)),
    ]:
        m = getattr(tf, cls)(getattr(tf, config)(**moe, **extra)).eval()
        batched_experts(m)
        yield label, Logits(m, lambda m, x: m(input_ids=x, use_cache=False).logits), ids(), ["i64"]

    # Speech (ADR-0027): the two HiFi-GAN vocoders (transposed 1D convolutions) and the Wav2Vec2
    # encoder (a convolutional positional embedding under weight normalization).
    hifi = dict(model_in_dim=16, upsample_initial_channel=32, upsample_rates=[4, 4],
                upsample_kernel_sizes=[8, 8], resblock_kernel_sizes=[3],
                resblock_dilation_sizes=[[1, 3]])
    vocoder = tf.SpeechT5HifiGan(tf.SpeechT5HifiGanConfig(**hifi)).eval()
    yield "SpeechT5 HiFi-GAN", Logits(vocoder, lambda m, x: m(x)), torch.randn(B, 20, 16), None
    vocoder = tf.FastSpeech2ConformerHifiGan(tf.FastSpeech2ConformerHifiGanConfig(**hifi)).eval()
    yield "FastSpeech2 HiFi-GAN", Logits(vocoder, lambda m, x: m(x)), torch.randn(B, 20, 16), None
    w2v = tf.Wav2Vec2Model(tf.Wav2Vec2Config(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        conv_dim=(16, 16), conv_stride=(5, 2), conv_kernel=(10, 3), num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2)).eval()
    yield "Wav2Vec2", Logits(w2v, lambda m, x: m(x).last_hidden_state), torch.randn(B, 1600), None

    class Lstm(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.rnn, self.out = nn.Embedding(V, 32), nn.LSTM(32, 32, batch_first=True), nn.Linear(32, V)

        def forward(self, x):
            return self.out(self.rnn(self.emb(x))[0])

    yield "LSTM", Lstm().eval(), ids(), ["i64"]

    class Gnn(nn.Module):
        """GCN-style message passing over a fixed graph (edges in a buffer)."""

        def __init__(self):
            super().__init__()
            self.register_buffer("src", torch.tensor([0, 1, 2, 3, 4, 5]))
            self.register_buffer("dst", torch.tensor([1, 2, 3, 4, 5, 0]))
            self.lin = nn.Linear(16, 16)

        def forward(self, x):  # x: [batch, 6 nodes, 16]
            msg = self.lin(x)[:, self.src]
            agg = torch.zeros_like(x).index_add(1, self.dst, msg)
            return torch.relu(agg + x)

    yield "GNN (index_add)", Gnn().eval(), torch.randn(B, 6, 16), None

    yield "MoE (top-2 routing)", Moe().eval(), torch.randn(B, T, 16), None

    class Conv1dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Conv1d(4, 8, 3, padding=1)
            self.n = nn.GroupNorm(2, 8)

        def forward(self, x):
            return torch.nn.functional.softplus(self.n(self.c(x))).mean(-1)

    yield "Conv1d + GroupNorm", Conv1dNet().eval(), torch.randn(B, 4, 16), None


GG = Path(os.environ.get("GG_BIN") or shutil.which("gg") or "gg-not-found")


def _gg(*args) -> str:
    out = subprocess.run([str(GG), *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip().splitlines()[-1] if out.stderr.strip() else "gg failed")
    return out.stdout


#: Batch sizes at which a captured description is executed, besides the capture's: a shape that
#: silently keeps the capture's number of examples fails here, not on the grid.
OTHER_BATCHES = (3, 5)


def resample(x, n: int):
    """`n` fresh examples shaped like `x`: token ids in the range `x` uses, or Gaussian values."""
    g = torch.Generator().manual_seed(1000 + n)
    if x.is_floating_point():
        return torch.randn((n,) + tuple(x.shape[1:]), generator=g)
    return torch.randint(0, int(x.max()) + 1, (n,) + tuple(x.shape[1:]), generator=g)


def verify(model, x, dtypes, workdir: Path):
    """Capture `model`, have the Rust tool validate and execute the description, and return
    `(nodes, operators, gap relative to PyTorch)`. The weights are perturbed first: a zero bias or
    a unit gain would hide a translation error. The description is executed at the capture batch
    **and** at other batch sizes (`OTHER_BATCHES`), each compared with PyTorch at that batch: the
    gap returned is the worst."""
    torch.manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    desc, weights = export_model(model, x, "mse", name="corpus", input_dtypes=dtypes)
    path = workdir / "m.json"
    path.write_text(json.dumps(desc))
    json.loads(_gg("job", "check-model", str(path), "--json"))
    ck = workdir / "ck.ggb"
    tensors.write_bundle(ck, weights)
    # The platform trains: BatchNorm there normalizes with batch statistics, never with running
    # averages (docs/PYTHON.md). The PyTorch reference is therefore taken in that mode.
    norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    gap = 0.0
    for xb in [x] + [resample(x, n) for n in OTHER_BATCHES if n != int(x.shape[0])]:
        inp, out = workdir / "x.ggt", workdir / "out.ggt"
        tensors.write_tensor(inp, xb.numpy().astype(np.float32))
        _gg("job", "eval-model", str(path), "--checkpoint", str(ck), "--input", f"x={inp}", "--out", str(out))
        got = tensors.read_tensor(out)
        for m in norms:
            m.train()
        with torch.no_grad():
            expected = model(xb).numpy()
        for m in norms:
            m.eval()
        if got.shape != expected.shape:
            raise RuntimeError(f"batch {xb.shape[0]}: shape {got.shape} versus {expected.shape}")
        gap = max(gap, float(np.abs(got - expected).max()) / max(1.0, float(np.abs(expected).max())))
    ops = sorted({n["op"] for n in desc["nodes"]})
    return len(desc["nodes"]), ops, gap


def main():
    rows = []
    blocking = Counter()
    gen = corpus()
    while True:
        try:
            label, model, x, dtypes = next(gen)
        except StopIteration:
            break
        r = coverage(model, x, input_dtypes=dtypes)
        if r["export_error"]:
            rows.append((label, "torch.export rejects", r["export_error"][:120]))
            continue
        if r["captured"]:
            if not GG.exists():
                rows.append((label, "**captured**", "not verified: gg binary missing"))
                continue
            try:
                with tempfile.TemporaryDirectory() as d:
                    nodes, _, gap = verify(model, x, dtypes, Path(d))
                rows.append((label, "**captured**", f"{nodes} nodes, gap to PyTorch {gap:.1e}"))
            except Exception as e:  # noqa: BLE001 — report it, do not hide it
                rows.append((label, "captured, **rejected**", str(e)[:160]))
            continue
        ops = sorted(r["missing"], key=lambda o: -r["missing"][o]["count"])
        for o in ops:
            blocking[o] += 1
        rows.append((label, "missing", ", ".join(f"`{o}` ×{r['missing'][o]['count']}" for o in ops)))
    print("| Model | Result | Detail |")
    print("|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} |")
    print()
    print("| Missing operator | Models blocked |")
    print("|---|---|")
    for op, n in blocking.most_common():
        print(f"| `{op}` | {n} |")
    captured = sum(1 for r in rows if r[1] == "**captured**")
    print(f"\n{captured} of {len(rows)} models captured.", file=sys.stderr)


if __name__ == "__main__":
    main()
