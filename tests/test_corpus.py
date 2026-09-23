"""Corpus of real models (tools/coverage_corpus.py, gg-core/docs/MODELS.md): every architecture the driver
captures must compute, once translated and executed by the Rust tool, **the same function** as
PyTorch.

`transformers` is not a dependency of the package: without it, these tests are skipped, which is
not a success. Every model of the corpus is captured since `gg-ops/8` (ADR-0029); the default
experts of `transformers`, whose shapes depend on the routing, are checked to be rejected with the
way out named.
"""

import sys
from functools import lru_cache
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import coverage_corpus  # noqa: E402

from gpugoblin import coverage  # noqa: E402

CAPTURED = ["Llama", "Qwen2", "Mistral", "Gemma", "Phi-3", "GPT-2", "GPT-NeoX", "BERT", "DistilBERT",
            "T5", "ViT", "ResNet", "ConvNeXt", "Whisper (encoder)", "Mamba", "Mixtral", "Qwen2-MoE",
            "OLMoE", "Qwen3-MoE", "SpeechT5 HiFi-GAN", "FastSpeech2 HiFi-GAN", "Wav2Vec2", "LSTM",
            "GNN (index_add)", "MoE (top-2 routing)", "Conv1d + GroupNorm"]


@lru_cache(maxsize=None)
def corpus():
    return {label: (model, x, dtypes) for label, model, x, dtypes in coverage_corpus.corpus()}


@pytest.mark.parametrize("label", CAPTURED)
def test_a_captured_model_computes_the_same_function(label, tmp_path):
    if not coverage_corpus.GG.exists():
        pytest.skip("gg binary not found (set GG_BIN)")
    model, x, dtypes = corpus()[label]
    nodes, ops, gap = coverage_corpus.verify(model, x, dtypes, tmp_path)
    print(f"{label}: {nodes} nodes, relative gap {gap:.1e}, {ops}")
    assert gap <= 1e-5, f"{label}: the Rust executor departs from PyTorch by {gap}"


@pytest.mark.parametrize("label", CAPTURED)
def test_the_description_has_the_parameters_of_the_module(label):
    """As many trainable elements in the description as in the module, counted by PyTorch, which
    counts a tied tensor once: an embedding shared with the language-model head is one parameter of
    the description, not two trained apart (GPT-2, Gemma, BERT, DistilBERT, T5 and Mamba had two
    before). The driver's synthetic constants (scalars, position tables, broadcast zeros) are not
    parameters of the module."""
    import numpy as np

    from gpugoblin import export_model
    from gpugoblin.export import constant_parameter_names

    model, x, dtypes = corpus()[label]
    desc, weights = export_model(model, x, "mse", input_dtypes=dtypes)
    synthetic = set(constant_parameter_names(desc))
    described = sum(int(np.prod(p["shape"])) for p in desc["parameters"] if p["name"] not in synthetic)
    assert described == sum(p.numel() for p in model.parameters()), label
    assert set(weights) == {p["name"] for p in desc["parameters"]}


def test_every_corpus_model_is_listed():
    """A model added to the corpus is checked here, captured or not: none is left out silently."""
    assert sorted(corpus()) == sorted(CAPTURED)


def test_the_default_experts_of_transformers_are_named_with_the_way_out():
    """Without `batched_experts`, transformers runs each expert on the tokens it received: shapes
    that depend on the routing. Capture says so, and says what to call."""
    import transformers as tf

    cfg = tf.MixtralConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                           num_key_value_heads=2, intermediate_size=32, num_local_experts=4,
                           num_experts_per_tok=2)
    m = coverage_corpus.Logits(tf.MixtralForCausalLM(cfg).eval(),
                               lambda m, x: m(input_ids=x, use_cache=False).logits)
    r = coverage(m, coverage_corpus.ids(), input_dtypes=["i64"])
    assert not r["captured"]
    assert any("batched_experts" in v["reason"] for v in r["missing"].values()), r["missing"]


def test_tied_weights_reload_into_the_module():
    """Mamba ties its language-model head to the embedding; the capture names it by the alias
    `lm_head`. The checkpoint must still reload, and the reloaded module computes what the
    original computed (observed on the grid on 2026-09-23: reloading failed)."""
    import numpy as np

    from gpugoblin import export_model
    from gpugoblin.export import constant_parameter_names, state_dict_from_bundle

    model, x, dtypes = corpus()["Mamba"]
    desc, weights = export_model(model, x, "mse", input_dtypes=dtypes)
    import copy

    other = copy.deepcopy(model)
    with torch.no_grad():
        for p in other.parameters():
            p.zero_()
    state_dict_from_bundle(other, {k: np.asarray(v) for k, v in weights.items()},
                           ignore=constant_parameter_names(desc))
    with torch.no_grad():
        assert torch.equal(other(x), model(x))
