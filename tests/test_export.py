import json
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from gpugoblin import UnsupportedOperator, export_model, tensors  # noqa: E402
from gpugoblin.export import state_dict_from_bundle  # noqa: E402
from gg_binary import GG  # noqa: E402


def check_with_gg(description, tmp_path):
    """The real checker (gg job check-model) accepts the description."""
    if not GG.exists():
        pytest.skip("gg binary not found (set GG_BIN)")
    p = tmp_path / "model.json"
    p.write_text(json.dumps(description))
    out = subprocess.run([str(GG), "job", "check-model", str(p), "--json"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_mlp_export(tmp_path):
    model = nn.Sequential(nn.Linear(6, 10), nn.ReLU(), nn.Linear(10, 3))
    desc, weights = export_model(model, torch.randn(4, 6), "cross_entropy", name="mlp")
    assert desc["format"] == "gg-graph/v1"
    assert desc["inputs"] == [{"name": "x", "shape": [0, 6], "dtype": "f32"}]
    assert sum(int(np.prod(p["shape"])) for p in desc["parameters"]) == sum(p.numel() for p in model.parameters())
    ops = [n["op"] for n in desc["nodes"]]
    assert "MatMul" in ops and "Relu" in ops and "Transpose" in ops
    report = check_with_gg(desc, tmp_path)
    assert report["parameters"] == 6 * 10 + 10 + 10 * 3 + 3
    # The weights keep the PyTorch layout and reload unchanged.
    bundle = tensors.decode_bundle(tensors.encode_bundle(weights))
    fresh = nn.Sequential(nn.Linear(6, 10), nn.ReLU(), nn.Linear(10, 3))
    state_dict_from_bundle(fresh, bundle)
    for a, b in zip(model.parameters(), fresh.parameters()):
        assert torch.equal(a, b)


def test_cnn_export(tmp_path):
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 4, 3), nn.Tanh(), nn.AvgPool2d(2), nn.Flatten(), nn.Linear(4 * 7 * 7, 10),
    )
    desc, _ = export_model(model, torch.randn(2, 3, 32, 32), "cross_entropy", name="cnn")
    ops = {n["op"] for n in desc["nodes"]}
    assert {"Conv2d", "BatchNorm2d", "MaxPool2d", "AvgPool2d", "Flatten", "Tanh"} <= ops
    # BatchNorm's running statistics are not graph parameters.
    assert all("running" not in p["name"] for p in desc["parameters"])
    report = check_with_gg(desc, tmp_path)
    assert report["shapes"]["flatten"] == "[N, 196]"


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, 16)
        self.norm = nn.LayerNorm(16)
        self.up = nn.Linear(16, 32)
        self.down = nn.Linear(32, 16)
        self.head = nn.Linear(16, 50)

    def forward(self, ids):
        h = self.emb(ids)
        h = h + self.down(torch.nn.functional.gelu(self.up(self.norm(h))))
        return self.head(h).reshape(-1, 50)


def test_embedding_model_export(tmp_path):
    model = TinyLM()
    desc, _ = export_model(model, torch.randint(0, 50, (3, 8)), "cross_entropy", name="lm",
                           input_dtypes=["i64"], target_shape=[0, 8])
    ops = [n["op"] for n in desc["nodes"]]
    assert ops[0] == "Gather" and "LayerNorm" in ops and "Gelu" in ops and "Add" in ops
    assert desc["inputs"][0]["dtype"] == "i64"
    report = check_with_gg(desc, tmp_path)
    assert report["shapes"][desc["loss"]["cross_entropy"]["prediction"]] == "[N, 50]"


def test_unsupported_operator_is_named():
    # A one-hot encoding is an integer tensor whose shape depends on a class count: it is outside
    # the profile, which says so by name (ADR-0029).
    class OneHot(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x):
            return torch.nn.functional.one_hot(self.lin(x).argmax(-1), 4).float()

    with pytest.raises(UnsupportedOperator) as e:
        export_model(OneHot(), torch.randn(2, 4), "mse")
    from gpugoblin.export import PROFILE_VERSION

    assert "one_hot" in str(e.value) and PROFILE_VERSION in str(e.value)

def test_tied_weights_are_one_parameter_and_reload_into_every_alias():
    """A language-model head that is the embedding matrix: one parameter in the description, one
    tensor in the weights; the checkpoint reloads into the module, whose two aliases stay one."""
    transformers = pytest.importorskip("transformers")
    from gpugoblin.export import constant_parameter_names, state_dict_from_bundle

    m = transformers.LlamaForCausalLM(transformers.LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, tie_word_embeddings=True)).eval()
    assert m.lm_head.weight is m.model.embed_tokens.weight

    class Logits(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(input_ids=x, use_cache=False).logits

    model = Logits().eval()
    ids = torch.randint(0, 64, (2, 5))
    desc, weights = export_model(model, ids, "mse", input_dtypes=["i64"])
    names = [p["name"] for p in desc["parameters"]]
    assert "m.model.embed_tokens.weight" in names and "m.lm_head.weight" not in names
    consumers = [n for n in desc["nodes"] if "m.model.embed_tokens.weight" in n["inputs"]]
    assert len(consumers) == 2, consumers  # the embedding lookup and the head
    counted = sum(int(np.prod(p["shape"])) for p in desc["parameters"]
                  if p["name"] not in constant_parameter_names(desc))
    assert counted == sum(p.numel() for p in model.parameters())
    # Reload: new values reach both aliases, which stay one tensor.
    new = {k: np.asarray(v) + 1.0 for k, v in weights.items()}
    state_dict_from_bundle(model, new, ignore=constant_parameter_names(desc))
    assert m.lm_head.weight is m.model.embed_tokens.weight
    assert torch.equal(m.lm_head.weight, torch.from_numpy(new["m.model.embed_tokens.weight"]))


def test_capture_gives_the_module_back_in_its_mode():
    """The capture evaluates the module in eval mode; the module comes back as it was."""
    m = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.1), nn.Linear(4, 2)).train()
    m[1].eval()  # a mixed state, as a fine-tuning with frozen statistics leaves it
    export_model(m, torch.randn(3, 4), "cross_entropy")
    assert m.training and not m[1].training and m[2].training
