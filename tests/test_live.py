"""End to end against a real server: enabled by GG_SERVER, GG_TOKEN and GG_PROJECT. With GG_NODE (a
lab node's local API), the data, the runs and the trained weights go through the node (ADR-0041)."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

import gpugoblin as gg  # noqa: E402


@pytest.mark.skipif(not os.environ.get("GG_SERVER"), reason="GG_SERVER not set")
def test_train_on_the_grid():
    """An ordinary PyTorch module trains on the grid and the loss goes down.

    This is the end-to-end proof: capture, description validated by the server, tasks executed by
    real workers, verified by a real validator, checkpoint reloaded into the module.
    """
    client = gg.Client(os.environ["GG_SERVER"], os.environ["GG_TOKEN"], node=os.environ.get("GG_NODE"))
    project = os.environ["GG_PROJECT"]
    profile = client.operator_profile()
    from gpugoblin.export import PROFILE_VERSION

    # The server publishes exactly the profile the driver targets.
    assert profile["profile"] == PROFILE_VERSION

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 24), nn.GELU(), nn.Linear(24, 3))
    # Separable problem: three classes determined by the first coordinate.
    rng = np.random.default_rng(1)
    n = 384
    y_np = rng.integers(0, 3, size=n)
    x_np = rng.standard_normal((n, 6)).astype(np.float32) * 0.3
    x_np[:, 0] += y_np * 2.0
    x, y = torch.from_numpy(x_np), torch.from_numpy(y_np).long()

    def loss_of(m):
        with torch.no_grad():
            return float(nn.functional.cross_entropy(m(x), y))

    before = loss_of(model)
    trainer = gg.Trainer(client, project, model, "cross_entropy", x[:2], name="live")
    assert trainer.parameter_count == sum(p.numel() for p in model.parameters())
    run = trainer.fit(x_np, y_np.astype(np.float32), rounds=25, batch_per_round=128,
                      tasks_per_round=4, lr=0.05)
    view = trainer.wait(run["run_id"], timeout=float(os.environ.get("GG_TIMEOUT", "600")))
    assert view["state"] == "completed", view
    assert view["step"] == 25
    assert len(view["checkpoint_steps"]) == 26

    trainer.load_checkpoint_into(model, run["run_id"])
    after = loss_of(model)
    print(f"loss before {before:.4f} -> after {after:.4f} ({view['step']} rounds on the grid)")
    assert after < before * 0.8, f"the loss did not go down: {before} -> {after}"


@pytest.mark.skipif(not os.environ.get("GG_SERVER"), reason="GG_SERVER not set")
def test_train_a_causal_language_model_on_the_grid():
    """A causal language model written like nanoGPT — triangular buffer, masked_fill with −∞,
    1/√d factor — trains on the grid to predict the next token, and the loss drops.

    The task is learnable: each sequence follows a fixed permutation of the tokens, so the next one
    follows from the current one. The mask guarantees that the model learns it instead of reading
    the answer.
    """
    import math

    import torch.nn.functional as F

    client = gg.Client(os.environ["GG_SERVER"], os.environ["GG_TOKEN"], node=os.environ.get("GG_NODE"))
    project = os.environ["GG_PROJECT"]
    vocab, seq, dim, heads = 17, 8, 24, 2

    class CausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, dim)
            self.norm = nn.LayerNorm(dim)
            self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
            self.proj = nn.Linear(dim, dim)
            self.head = nn.Linear(dim, vocab)
            self.register_buffer("bias", torch.tril(torch.ones(seq, seq)).view(1, 1, seq, seq))

        def forward(self, ids):
            b, t = ids.shape
            x = self.emb(ids)
            n = self.norm(x)
            split = lambda z: z.view(b, t, heads, dim // heads).transpose(1, 2)  # noqa: E731
            q, k, v = split(self.q(n)), split(self.k(n)), split(self.v(n))
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :t, :t] == 0, float("-inf"))
            y = (F.softmax(att, dim=-1) @ v).transpose(1, 2).contiguous().view(b, t, dim)
            x = x + self.proj(y)
            return self.head(x).reshape(-1, vocab)

    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    perm = rng.permutation(vocab)
    n = 512
    ids = np.zeros((n, seq), dtype=np.int64)
    ids[:, 0] = rng.integers(0, vocab, size=n)
    for t in range(1, seq):
        ids[:, t] = perm[ids[:, t - 1]]
    targets = perm[ids]  # the next token at each position
    x, y = torch.from_numpy(ids), torch.from_numpy(targets)

    model = CausalLM()

    def loss_of(m):
        with torch.no_grad():
            return float(F.cross_entropy(m(x), y.reshape(-1)))

    before = loss_of(model)
    trainer = gg.Trainer(client, project, model, "cross_entropy", x[:2], name="causal-lm",
                         input_dtypes=["i64"], target_shape=[0, seq])
    assert [nd["op"] for nd in trainer.description["nodes"]].count("CausalMask") == 1
    run = trainer.fit(ids.astype(np.float32), targets.astype(np.float32), rounds=40,
                      batch_per_round=128, tasks_per_round=4, lr=0.03)
    view = trainer.wait(run["run_id"], timeout=float(os.environ.get("GG_TIMEOUT", "900")))
    assert view["state"] == "completed", view
    trainer.load_checkpoint_into(model, run["run_id"])
    after = loss_of(model)
    with torch.no_grad():
        accuracy = float((model(x).argmax(-1) == y.reshape(-1)).float().mean())
    print(f"causal language model: loss {before:.4f} -> {after:.4f} (ln V = {math.log(vocab):.4f}), "
          f"accuracy {accuracy:.1%}, {view['step']} rounds on the grid")
    assert after < 0.3 * math.log(vocab), f"the loss did not drop: {before} -> {after}"


@pytest.mark.skipif(not os.environ.get("GG_SERVER"), reason="GG_SERVER not set")
def test_train_a_rotary_model_on_an_order_task_on_the_grid():
    """A RoPE model written Hugging Face-style learns, on the grid, a task that depends on order:
    copy the token two positions earlier. Without position encoding, a causal model only partly
    manages it (docs/adr/0021)."""
    import math

    import torch.nn.functional as F

    client = gg.Client(os.environ["GG_SERVER"], os.environ["GG_TOKEN"], node=os.environ.get("GG_NODE"))
    project = os.environ["GG_PROJECT"]
    vocab, seq, dim, heads = 16, 8, 32, 2

    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    class RopeLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, dim)
            self.norm = nn.LayerNorm(dim)
            self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
            self.proj = nn.Linear(dim, dim)
            self.head = nn.Linear(dim, vocab)
            hd = dim // heads
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, hd, 2).float() / hd))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        def forward(self, ids):
            b, t = ids.shape
            x = self.emb(ids)
            n = self.norm(x)
            split = lambda z: z.view(b, t, heads, dim // heads).transpose(1, 2)  # noqa: E731
            q, k, v = split(self.q(n)), split(self.k(n)), split(self.v(n))
            freqs = torch.outer(torch.arange(t, dtype=torch.float32), self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos()[None, None], emb.sin()[None, None]
            q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.proj(y.transpose(1, 2).contiguous().view(b, t, dim))
            return self.head(x).reshape(-1, vocab)

    torch.manual_seed(0)
    rng = np.random.default_rng(5)
    n = 1024
    ids = rng.integers(0, vocab, size=(n, seq))
    targets = np.zeros_like(ids)
    targets[:, 2:] = ids[:, :-2]  # the token two positions earlier
    x, y = torch.from_numpy(ids), torch.from_numpy(targets)
    model = RopeLM()

    def measure(m):
        with torch.no_grad():
            logits = m(x)
            loss = float(F.cross_entropy(logits, y.reshape(-1)))
            pred = logits.argmax(-1).reshape(n, seq)
            acc = float((pred[:, 2:] == y[:, 2:]).float().mean())
        return loss, acc

    before, acc_before = measure(model)
    trainer = gg.Trainer(client, project, model, "cross_entropy", x[:2], name="rope-copy",
                         input_dtypes=["i64"], target_shape=[0, seq])
    ops = [nd["op"] for nd in trainer.description["nodes"]]
    assert "Slice" in ops and "CausalMask" in ops
    inits = [p["init"] for p in trainer.description["parameters"] if isinstance(p["init"], dict)]
    assert any("rotary_cos" in i for i in inits) and any("rotary_sin" in i for i in inits)
    run = trainer.fit(ids.astype(np.float32), targets.astype(np.float32), rounds=60,
                      batch_per_round=128, tasks_per_round=4, lr=0.01)
    view = trainer.wait(run["run_id"], timeout=float(os.environ.get("GG_TIMEOUT", "1200")))
    assert view["state"] == "completed", view
    trainer.load_checkpoint_into(model, run["run_id"])
    after, acc_after = measure(model)
    print(f"RoPE, copy at t−2: loss {before:.4f} -> {after:.4f} (ln V = {math.log(vocab):.4f}), "
          f"accuracy on positions ≥ 2: {acc_before:.1%} -> {acc_after:.1%}, {view['step']} rounds")
    # Chance gives 1/16 = 6%; measured on 2026-09-21: 92%. The margin covers other hardware.
    assert acc_after > 0.8, f"the order task was not learned: {acc_after:.1%}"


@pytest.mark.skipif(not os.environ.get("GG_SERVER"), reason="GG_SERVER not set")
def test_train_a_scaled_rotary_model_on_the_grid():
    """The real rotary class from `transformers`, scaled with YaRN, trains on the grid: explicit
    frequencies and an amplitude other than 1, recognized by value, executed by real workers and
    verified by a real validator (docs/adr/0022)."""
    import logging
    import math

    import torch.nn.functional as F

    pytest.importorskip("transformers")
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

    logging.getLogger("transformers").setLevel(logging.ERROR)
    client = gg.Client(os.environ["GG_SERVER"], os.environ["GG_TOKEN"], node=os.environ.get("GG_NODE"))
    project = os.environ["GG_PROJECT"]
    vocab, seq, dim, heads = 16, 8, 32, 2

    class YarnLM(nn.Module):
        def __init__(self):
            super().__init__()
            # Original context of 4 positions extended to 16: the sequence of 8 is in the scaled
            # range, the low frequencies are interpolated, the amplitude is 1.139.
            config = LlamaConfig(hidden_size=dim, num_attention_heads=heads, max_position_embeddings=16,
                                 rope_parameters={"rope_type": "yarn", "rope_theta": 10000.0,
                                                  "factor": 4.0, "original_max_position_embeddings": 4})
            self.rope = LlamaRotaryEmbedding(config)
            self.emb = nn.Embedding(vocab, dim)
            self.norm = nn.LayerNorm(dim)
            self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
            self.proj = nn.Linear(dim, dim)
            self.head = nn.Linear(dim, vocab)

        def forward(self, ids):
            b, t = ids.shape
            x = self.emb(ids)
            n = self.norm(x)
            split = lambda z: z.view(b, t, heads, dim // heads).transpose(1, 2)  # noqa: E731
            q, k, v = split(self.q(n)), split(self.k(n)), split(self.v(n))
            cos, sin = self.rope(x, torch.arange(t)[None])
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.proj(y.transpose(1, 2).contiguous().view(b, t, dim))
            return self.head(x).reshape(-1, vocab)

    torch.manual_seed(0)
    rng = np.random.default_rng(5)
    n = 1024
    ids = rng.integers(0, vocab, size=(n, seq))
    targets = np.zeros_like(ids)
    targets[:, 2:] = ids[:, :-2]  # the token two positions earlier
    x, y = torch.from_numpy(ids), torch.from_numpy(targets)
    model = YarnLM()
    assert abs(model.rope.attention_scaling - (0.1 * math.log(4.0) + 1.0)) < 1e-12

    def measure(m):
        with torch.no_grad():
            logits = m(x)
            loss = float(F.cross_entropy(logits, y.reshape(-1)))
            pred = logits.argmax(-1).reshape(n, seq)
            acc = float((pred[:, 2:] == y[:, 2:]).float().mean())
        return loss, acc

    before, acc_before = measure(model)
    trainer = gg.Trainer(client, project, model, "cross_entropy", x[:2], name="yarn-copy",
                         input_dtypes=["i64"], target_shape=[0, seq])
    laws = [next(iter(p["init"].values())) for p in trainer.description["parameters"]
            if isinstance(p["init"], dict) and ("rotary_cos" in p["init"] or "rotary_sin" in p["init"])]
    assert len(laws) == 2 and laws[0] == laws[1], laws
    assert "frequencies" in laws[0] and laws[0]["amplitude"] > 1.13, laws[0]
    run = trainer.fit(ids.astype(np.float32), targets.astype(np.float32), rounds=60,
                      batch_per_round=128, tasks_per_round=4, lr=0.01)
    view = trainer.wait(run["run_id"], timeout=float(os.environ.get("GG_TIMEOUT", "1200")))
    assert view["state"] == "completed", view
    trainer.load_checkpoint_into(model, run["run_id"])
    after, acc_after = measure(model)
    print(f"YaRN (amplitude {laws[0]['amplitude']:.4f}), copy at t−2: loss {before:.4f} -> "
          f"{after:.4f} (ln V = {math.log(vocab):.4f}), accuracy on positions ≥ 2: "
          f"{acc_before:.1%} -> {acc_after:.1%}, {view['step']} rounds")
    assert acc_after > 0.8, f"the order task was not learned: {acc_after:.1%}"


@pytest.mark.skipif(not os.environ.get("GG_SERVER"), reason="GG_SERVER not set")
def test_train_a_real_transformers_llama_on_the_grid():
    """A `LlamaForCausalLM` from `transformers`, as is — grouped queries, SwiGLU, hand-written
    RMSNorm, Llama 3.1 RoPE — trains on the grid through the driver, without a single line adapted
    (gg-server/docs/ROADMAP.md, P11 C3). Task: copy the token two positions earlier."""
    import logging
    import math

    import torch.nn.functional as F

    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    logging.getLogger("transformers").setLevel(logging.ERROR)
    client = gg.Client(os.environ["GG_SERVER"], os.environ["GG_TOKEN"], node=os.environ.get("GG_NODE"))
    project = os.environ["GG_PROJECT"]
    vocab, seq = 16, 8

    class Logits(nn.Module):
        def __init__(self):
            super().__init__()
            rope = {"rope_type": "llama3", "rope_theta": 500000.0, "factor": 8.0,
                    "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                    "original_max_position_embeddings": 4}
            self.lm = LlamaForCausalLM(LlamaConfig(
                vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
                rope_parameters=rope))

        def forward(self, ids):
            return self.lm(input_ids=ids, use_cache=False).logits.reshape(-1, vocab)

    torch.manual_seed(0)
    rng = np.random.default_rng(5)
    n = 1024
    ids = rng.integers(0, vocab, size=(n, seq))
    targets = np.zeros_like(ids)
    targets[:, 2:] = ids[:, :-2]
    x, y = torch.from_numpy(ids), torch.from_numpy(targets)
    model = Logits()

    def measure(m):
        with torch.no_grad():
            logits = m(x)
            loss = float(F.cross_entropy(logits, y.reshape(-1)))
            pred = logits.argmax(-1).reshape(n, seq)
            acc = float((pred[:, 2:] == y[:, 2:]).float().mean())
        return loss, acc

    before, acc_before = measure(model)
    trainer = gg.Trainer(client, project, model, "cross_entropy", x[:2], name="llama-hf",
                         input_dtypes=["i64"], target_shape=[0, seq])
    ops = {nd["op"] for nd in trainer.description["nodes"]}
    assert {"ReduceMean", "Rsqrt", "Concat", "CausalMask"} <= ops, ops
    # 300 rounds: trained alone in PyTorch with the same settings, this model reaches 35% after 60
    # steps and 98% after 300 (measured on 2026-09-23); the model needs these steps, not the grid.
    run = trainer.fit(ids.astype(np.float32), targets.astype(np.float32), rounds=300,
                      batch_per_round=128, tasks_per_round=4, lr=0.01)
    view = trainer.wait(run["run_id"], timeout=float(os.environ.get("GG_TIMEOUT", "3600")))
    assert view["state"] == "completed", view
    trainer.load_checkpoint_into(model, run["run_id"])
    after, acc_after = measure(model)
    print(f"transformers Llama (GQA, SwiGLU, Llama 3.1 RoPE), copy at t−2: loss {before:.4f} -> "
          f"{after:.4f} (ln V = {math.log(vocab):.4f}), accuracy {acc_before:.1%} -> {acc_after:.1%}, "
          f"{view['step']} rounds")
    assert acc_after > 0.8, f"the order task was not learned: {acc_after:.1%}"


def _copy_task(vocab, seq, n=1024):
    """Copy the token two positions earlier; accuracy is measured from t = 2 onward."""
    rng = np.random.default_rng(5)
    ids = rng.integers(0, vocab, size=(n, seq))
    targets = np.zeros_like(ids)
    targets[:, 2:] = ids[:, :-2]
    return ids, targets


def _train_copy(model, name, rounds, lr=0.01):
    import torch.nn.functional as F

    client = gg.Client(os.environ["GG_SERVER"], os.environ["GG_TOKEN"], node=os.environ.get("GG_NODE"))
    vocab, seq = 16, 8
    ids, targets = _copy_task(vocab, seq)
    x, y = torch.from_numpy(ids), torch.from_numpy(targets)

    def measure(m):
        with torch.no_grad():
            logits = m(x)
            loss = float(F.cross_entropy(logits, y.reshape(-1)))
            acc = float((logits.argmax(-1).reshape(x.shape)[:, 2:] == y[:, 2:]).float().mean())
        return loss, acc

    before, acc_before = measure(model)
    trainer = gg.Trainer(client, os.environ["GG_PROJECT"], model, "cross_entropy", x[:2], name=name,
                         input_dtypes=["i64"], target_shape=[0, seq])
    run = trainer.fit(ids.astype(np.float32), targets.astype(np.float32), rounds=rounds,
                      batch_per_round=128, tasks_per_round=4, lr=lr)
    view = trainer.wait(run["run_id"], timeout=float(os.environ.get("GG_TIMEOUT", "3600")))
    assert view["state"] == "completed", view
    trainer.load_checkpoint_into(model, run["run_id"])
    after, acc_after = measure(model)
    ops = sorted({nd["op"] for nd in trainer.description["nodes"]})
    print(f"{name} ({len(trainer.description['nodes'])} nodes, {ops}): loss {before:.4f} -> {after:.4f}, "
          f"accuracy {acc_before:.1%} -> {acc_after:.1%}, {view['step']} rounds")
    return acc_after


@pytest.mark.skipif(not os.environ.get("GG_SERVER"), reason="GG_SERVER not set")
def test_train_an_lstm_with_a_custom_operator_on_the_grid():
    """An `nn.LSTM` (unrolled over time by the driver) followed by a custom activation written with
    `gpugoblin.formula`: two things the profile could not express before `gg-ops/6`.
    Calibrated alone in PyTorch with the same settings: 70% after 60 steps, 100% after 150
    (2026-09-23)."""
    vocab = 16

    class Lstm(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.rnn, self.out = nn.Embedding(vocab, 32), nn.LSTM(32, 48, batch_first=True), nn.Linear(48, vocab)

        def forward(self, ids):
            h = self.rnn(self.emb(ids))[0]
            return self.out(gg.formula("a * tanh(softplus(a))", h)).reshape(-1, vocab)

    torch.manual_seed(0)
    acc = _train_copy(Lstm(), "lstm-formula", rounds=150)
    assert acc > 0.9, f"the copy task was not learned: {acc:.1%}"


@pytest.mark.skipif(not os.environ.get("GG_SERVER"), reason="GG_SERVER not set")
def test_train_a_real_transformers_mamba_on_the_grid():
    """A `MambaForCausalLM` from `transformers` as is — causal 1D convolution, unrolled selective
    state scan, softplus, exp — which stumbled on ten operators before `gg-ops/6`. Calibrated alone
    in PyTorch: 100% after 30 steps (2026-09-23)."""
    import logging

    pytest.importorskip("transformers")
    from transformers import MambaConfig, MambaForCausalLM

    logging.getLogger("transformers").setLevel(logging.ERROR)
    vocab = 16

    class Logits(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm = MambaForCausalLM(MambaConfig(vocab_size=vocab, hidden_size=32, state_size=8,
                                                   num_hidden_layers=2, expand=2, conv_kernel=4))

        def forward(self, ids):
            return self.lm(input_ids=ids, use_cache=False).logits.reshape(-1, vocab)

    torch.manual_seed(0)
    acc = _train_copy(Logits(), "mamba-hf", rounds=60)
    assert acc > 0.9, f"the copy task was not learned: {acc:.1%}"


@pytest.mark.skipif(not os.environ.get("GG_STAGED_DEVICE"), reason="GG_STAGED_DEVICE not set")
@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_a_model_larger_than_every_worker_trains_in_stages_on_the_grid(tied):
    """A `transformers` Llama larger than each of the workers: they advertise a memory
    (`GG_STAGED_DEVICE`) smaller than what a task on the whole model requires. The driver writes
    nothing more: the grid's plan says that no machine can hold it whole, proposes stages, and
    `fit()` uses them (gg-server/docs/ROADMAP.md, D2 and D4). Tied: the language-model head is the embedding
    matrix, one parameter used by the first and the last stage, whose partial gradients the grid
    sums."""
    import logging
    import math

    import torch.nn.functional as F

    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    logging.getLogger("transformers").setLevel(logging.ERROR)
    client = gg.Client(os.environ["GG_SERVER"], os.environ["GG_TOKEN"], node=os.environ.get("GG_NODE"))
    project = os.environ["GG_PROJECT"]
    device = int(os.environ["GG_STAGED_DEVICE"])
    vocab, seq = 16, 8

    class Logits(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm = LlamaForCausalLM(LlamaConfig(
                vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
                tie_word_embeddings=tied))

        def forward(self, ids):
            return self.lm(input_ids=ids, use_cache=False).logits.reshape(-1, vocab)

    torch.manual_seed(0)
    rng = np.random.default_rng(5)
    n = 1024
    ids = rng.integers(0, vocab, size=(n, seq))
    targets = np.zeros_like(ids)
    targets[:, 2:] = ids[:, :-2]
    x, y = torch.from_numpy(ids), torch.from_numpy(targets)
    model = Logits()

    def measure(m):
        with torch.no_grad():
            logits = m(x)
            loss = float(F.cross_entropy(logits, y.reshape(-1)))
            acc = float((logits.argmax(-1).reshape(n, seq)[:, 2:] == y[:, 2:]).float().mean())
        return loss, acc

    before, acc_before = measure(model)
    trainer = gg.Trainer(client, project, model, "cross_entropy", x[:2], name="llama-stages",
                         input_dtypes=["i64"], target_shape=[0, seq])
    shared = "lm.model.embed_tokens.weight"
    names = [p["name"] for p in trainer.description["parameters"]]
    assert (shared in names) and (("lm.lm_head.weight" in names) is not tied), names
    plan = trainer.plan(batch_per_round=128, tasks_per_round=4)
    print("plan:", plan["advice"], "; whole task", plan["plain_task_device_bytes"],
          "bytes, workers at", device)
    assert plan["plain_task_device_bytes"] > device
    assert plan["recommendation"] is None and plan["staged"] is not None, plan
    assert plan["staged"]["largest_stage_bytes"] <= device
    rounds = int(os.environ.get("GG_STAGED_ROUNDS", "40"))
    run = trainer.fit(ids.astype(np.float32), targets.astype(np.float32), rounds=rounds,
                      batch_per_round=128, tasks_per_round=4, lr=0.01)
    bounds = run["config"]["graph"]["stage_bounds"]
    assert bounds, run["config"]["graph"]
    if tied:
        # The stages that hold the shared matrix, from the execution order of the description.
        import json
        import subprocess
        import tempfile

        from gg_binary import GG

        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(trainer.description, f)
            f.flush()
            order = json.loads(subprocess.run([str(GG), "job", "check-model", f.name, "--json"],
                                              capture_output=True, text=True, check=True).stdout)["order"]
        users = {i for i, o in enumerate(order)
                 for nd in trainer.description["nodes"] if nd["output"] == o and shared in nd["inputs"]}
        holders = {k for k in range(len(bounds) - 1) for i in users if bounds[k] <= i < bounds[k + 1]}
        print(f"shared matrix used by stages {sorted(holders)} of {len(bounds) - 1}")
        assert len(holders) >= 2, (bounds, sorted(users))
    view = trainer.wait(run["run_id"], timeout=float(os.environ.get("GG_TIMEOUT", "3600")))
    assert view["state"] == "completed", view
    trainer.load_checkpoint_into(model, run["run_id"])
    after, acc_after = measure(model)
    stages = len(run["config"]["graph"]["stage_bounds"]) - 1
    if tied:
        assert model.lm.lm_head.weight is model.lm.model.embed_tokens.weight
    print(f"{'tied ' if tied else ''}Llama in {stages} stages on workers of {device} bytes: loss {before:.4f} -> "
          f"{after:.4f} (ln V = {math.log(vocab):.4f}), accuracy {acc_before:.1%} -> {acc_after:.1%}, "
          f"{view['step']} rounds")
    assert after < before - 0.3, f"the loss did not go down: {before} -> {after}"


@pytest.mark.skipif(not os.environ.get("GG_TILED_DEVICE"), reason="GG_TILED_DEVICE not set")
def test_a_matrix_larger_than_every_worker_is_split_on_the_grid():
    """A matrix larger than the memory of each of the workers (`GG_TILED_DEVICE`, 3 MB): the grid's
    plan proposes to tile it, spreading the tiles across stages; `fit()` applies this without asking
    anything, and `load_checkpoint_into` reassembles the tiles in the module, which gets its whole
    matrices back (gg-server/docs/ROADMAP.md, D3 and D4). Task: predict the next token of a sequence
    `t, t+1, t+2…` modulo the vocabulary."""
    import torch.nn.functional as F

    client = gg.Client(os.environ["GG_SERVER"], os.environ["GG_TOKEN"], node=os.environ.get("GG_NODE"))
    project = os.environ["GG_PROJECT"]
    device = int(os.environ["GG_TILED_DEVICE"])
    # The large matrix is internal (`mid`, 1024 × 1024, 4 MiB); vocabulary and activations stay
    # small: it is the matrix, not the micro-batch, that exceeds a worker's memory.
    vocab, dim, seq = 64, 1024, 8

    class Next(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab, dim)
            self.mid = nn.Linear(dim, dim)
            self.head = nn.Linear(dim, vocab)

        def forward(self, ids):
            return self.head(F.gelu(self.mid(self.emb(ids)))).reshape(-1, vocab)

    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    n = 512
    start = rng.integers(0, vocab, size=(n, 1))
    ids = (start + np.arange(seq)[None, :]) % vocab
    targets = (ids + 1) % vocab
    x, y = torch.from_numpy(ids), torch.from_numpy(targets)
    model = Next()
    biggest = max(p.numel() * 4 for p in model.parameters())
    assert biggest > device, "the largest matrix must exceed the workers' memory"

    def loss_of(m):
        with torch.no_grad():
            return float(F.cross_entropy(m(x), y.reshape(-1)))

    before = loss_of(model)
    trainer = gg.Trainer(client, project, model, "cross_entropy", x[:2], name="tiles",
                         input_dtypes=["i64"], target_shape=[0, seq])
    # Micro-batches of 4 examples: the matrices, not the activations, are what exceed.
    plan = trainer.plan(batch_per_round=32, tasks_per_round=8)
    print("plan:", plan["advice"])
    assert plan["staged"] is not None and plan["staged"].get("tile_max_param_bytes"), plan
    run = trainer.fit(ids.astype(np.float32), targets.astype(np.float32),
                      rounds=int(os.environ.get("GG_TILED_ROUNDS", "30")),
                      batch_per_round=32, tasks_per_round=8, lr=0.05)
    tiling = run["config"]["graph"]["tiling"]
    assert tiling, run["config"]["graph"]
    view = trainer.wait(run["run_id"], timeout=float(os.environ.get("GG_TIMEOUT", "3600")))
    assert view["state"] == "completed", view
    trainer.load_checkpoint_into(model, run["run_id"])
    # The tiles are reassembled: the module has its whole matrices back.
    assert model.mid.weight.shape == (dim, dim)
    assert model.emb.weight.shape == (vocab, dim) and model.head.weight.shape == (vocab, dim)
    after = loss_of(model)
    print(f"matrices of {biggest} bytes on workers of {device}: {len(tiling)} matrix(es) "
          f"tiled, {len(run['config']['graph']['stage_bounds']) - 1} stages, loss "
          f"{before:.4f} -> {after:.4f}, {view['step']} rounds")
    assert after < before - 0.3, f"the loss did not go down: {before} -> {after}"


@pytest.mark.skipif(not os.environ.get("GG_SERVER"), reason="GG_SERVER not set")
def test_a_mixture_of_experts_trains_on_the_grid():
    """A `transformers` Mixtral (4 experts, top-2 routing) trains on the grid: the router's choice
    is computed inside the graph as integer tensors (argmax/top-k, gathers), on the workers, and
    recomputed by the validator (ADR-0029). Task: copy the token two positions earlier."""
    import logging
    import math

    import torch.nn.functional as F

    pytest.importorskip("transformers")
    from transformers import MixtralConfig, MixtralForCausalLM

    logging.getLogger("transformers").setLevel(logging.ERROR)
    client = gg.Client(os.environ["GG_SERVER"], os.environ["GG_TOKEN"], node=os.environ.get("GG_NODE"))
    project = os.environ["GG_PROJECT"]
    vocab, seq = 16, 8

    class Logits(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm = MixtralForCausalLM(MixtralConfig(
                vocab_size=vocab, hidden_size=32, intermediate_size=32, num_hidden_layers=1,
                num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
                num_local_experts=4, num_experts_per_tok=2))

        def forward(self, ids):
            return self.lm(input_ids=ids, use_cache=False).logits.reshape(-1, vocab)

    torch.manual_seed(0)
    ids, targets = _copy_task(vocab, seq)
    x, y = torch.from_numpy(ids), torch.from_numpy(targets)
    model = Logits()
    assert gg.batched_experts(model), "the experts must switch to static shapes"

    def measure(m):
        with torch.no_grad():
            logits = m(x)
            loss = float(F.cross_entropy(logits, y.reshape(-1)))
            pred = logits.argmax(-1).reshape(len(ids), seq)
            acc = float((pred[:, 2:] == y[:, 2:]).float().mean())
        return loss, acc

    before, acc_before = measure(model)
    trainer = gg.Trainer(client, project, model, "cross_entropy", x[:2], name="mixtral-mini",
                         input_dtypes=["i64"], target_shape=[0, seq])
    ops = {nd["op"] for nd in trainer.description["nodes"]}
    assert {"ArgTopK", "GatherElements"} <= ops, ops
    rounds = int(os.environ.get("GG_MOE_ROUNDS", "300"))
    run = trainer.fit(ids.astype(np.float32), targets.astype(np.float32), rounds=rounds,
                      batch_per_round=128, tasks_per_round=4, lr=0.01)
    view = trainer.wait(run["run_id"], timeout=float(os.environ.get("GG_TIMEOUT", "3600")))
    assert view["state"] == "completed", view
    trainer.load_checkpoint_into(model, run["run_id"])
    after, acc_after = measure(model)
    print(f"transformers Mixtral (4 experts, top-2), copy at t−2: loss {before:.4f} -> {after:.4f} "
          f"(ln V = {math.log(vocab):.4f}), accuracy {acc_before:.1%} -> {acc_after:.1%}, "
          f"{view['step']} rounds")
    assert acc_after > 0.8, f"the copy task was not learned: {acc_after:.1%}"
