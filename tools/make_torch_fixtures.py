"""Fixtures for the Rust tests (ADR-0027): real `transformers` speech models, captured, with the
loss and the gradient PyTorch computes for them, at a batch (3) other than the capture's (2).

For each model: the description (`<name>.json`), a `GGB1` bundle (`<name>.torch.ggb`) holding the
weights under their gg names, the input `__x`, the target `__y` and PyTorch's loss `__loss`, and a
second one (`<name>.grad.ggb`) holding PyTorch's gradient of every trained parameter, under its gg
name. The Rust test
(`crates/gg-compute/tests/graph_ops7.rs`) computes the same step with the executor and compares:
this is what says that a captured model **trains** as in PyTorch, not only that it evaluates the
same function.

    python tools/make_torch_fixtures.py              # rewrites crates/gg-compute/tests/fixtures/
    python tools/make_torch_fixtures.py convnext-mini  # only the named ones
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn

import coverage_corpus
from gpugoblin import batched_experts, export_model, tensors
from gpugoblin.export import _clean_name, tied_parameters

logging.disable(logging.WARNING)
FIXTURES = Path(__file__).resolve().parents[2] / "crates" / "gg-compute" / "tests" / "fixtures"


class Out(nn.Module):
    def __init__(self, inner, call):
        super().__init__()
        self.inner, self.call = inner, call

    def forward(self, x):
        return self.call(self.inner, x)


def models():
    import transformers as tf

    torch.manual_seed(0)
    hifi = dict(model_in_dim=8, upsample_initial_channel=16, upsample_rates=[4, 2],
                upsample_kernel_sizes=[8, 4], resblock_kernel_sizes=[3],
                resblock_dilation_sizes=[[1, 3]])
    yield "hifigan-mini", Out(tf.SpeechT5HifiGan(tf.SpeechT5HifiGanConfig(**hifi)), lambda m, x: m(x)), \
        torch.randn(3, 6, 8)
    w2v = tf.Wav2Vec2Config(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                            intermediate_size=32, conv_dim=(8, 8), conv_stride=(5, 2),
                            conv_kernel=(10, 3), num_conv_pos_embeddings=8,
                            num_conv_pos_embedding_groups=2)
    yield "wav2vec2-mini", Out(tf.Wav2Vec2Model(w2v), lambda m, x: m(x).last_hidden_state), \
        torch.randn(3, 400)
    # Mixtures of experts (ADR-0029): Mixtral through transformers' batched_mm experts, and the
    # hand-written top-2 gating of the corpus.
    mix = tf.MixtralConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                           num_key_value_heads=1, intermediate_size=16, num_local_experts=4,
                           num_experts_per_tok=2, max_position_embeddings=16)
    model = tf.MixtralForCausalLM(mix)
    batched_experts(model)
    yield "mixtral-mini", Out(model, lambda m, x: m(input_ids=x, use_cache=False).logits), \
        torch.randint(0, 32, (3, 6))
    yield "moe-top2-mini", coverage_corpus.Moe(), torch.randn(3, 5, 16)
    # Tied weights: the language-model head is the embedding matrix. One gg parameter, whose
    # gradient sums both uses.
    tied = tf.LlamaForCausalLM(tf.LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=True))
    assert tied.lm_head.weight is tied.model.embed_tokens.weight
    yield "llama-tied-mini", Out(tied, lambda m, x: m(input_ids=x, use_cache=False).logits), \
        torch.randint(0, 64, (3, 6))
    # Depthwise 7×7 convolutions (one group per channel): the executor computes their weight
    # gradient by another route than Burn's (exec/extra.rs, `depthwise_conv2d`).
    cnx = tf.ConvNextConfig(num_channels=3, num_stages=2, hidden_sizes=[8, 16], depths=[1, 1], image_size=32)
    yield "convnext-mini", Out(tf.ConvNextModel(cnx), lambda m, x: m(pixel_values=x).pooler_output), \
        torch.randn(3, 3, 32, 32)


def main():
    import sys
    FIXTURES.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    for name, model, x in models():
        if only and name not in only:
            continue
        model = model.eval()
        torch.manual_seed(1)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.05 * torch.randn_like(p))
        dtypes = ["i64"] if not x.is_floating_point() else None
        # Captured on two examples, checked on three: the description must hold at a batch size
        # other than the capture's (a reshape that kept the capture's count would fail here).
        desc, weights = export_model(model, x[:2], "mse", name=name, input_dtypes=dtypes)
        trained = {p["name"] for p in desc["parameters"] if p.get("trainable", True)}
        out = model(x)
        y = torch.randn_like(out)
        loss = torch.nn.functional.mse_loss(out, y)
        model.zero_grad()
        loss.backward()
        bundle = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        # gg names in capture order, as the driver gives them (see `state_dict_from_bundle`).
        taken: set = set()
        matched = set()
        grads = {}
        tied = tied_parameters(model)
        for key, p in model.named_parameters(remove_duplicate=False):
            if tied[key] != key:
                continue  # an alias of a tied parameter: the same gg parameter, the same gradient
            gg = _clean_name(key, taken)
            if gg in trained:
                # A parameter that does not take part in the forward pass has a zero gradient.
                g = p.grad if p.grad is not None else torch.zeros_like(p)
                grads[gg] = g.detach().numpy().astype(np.float32)
                matched.add(gg)
        missing = trained - matched
        if missing:
            raise SystemExit(f"{name}: no PyTorch gradient for {sorted(missing)[:5]}")
        bundle["__x"] = x.numpy().astype(np.float32)  # integer ids are exact in f32
        bundle["__y"] = y.detach().numpy().astype(np.float32)
        bundle["__loss"] = np.array([float(loss.detach())], dtype=np.float32)
        (FIXTURES / f"{name}.json").write_text(json.dumps(desc, indent=1) + "\n")
        tensors.write_bundle(FIXTURES / f"{name}.torch.ggb", bundle)
        tensors.write_bundle(FIXTURES / f"{name}.grad.ggb", grads)
        size = sum((FIXTURES / f"{name}.{k}.ggb").stat().st_size for k in ("torch", "grad"))
        print(f"{name}: {len(desc['nodes'])} nodes, {len(trained)} trained parameters, "
              f"loss {float(loss.detach()):.6f}, bundles {size} bytes")


if __name__ == "__main__":
    main()
