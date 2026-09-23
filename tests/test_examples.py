"""The examples run as documented. The offline one needs only `gg`; the others need a grid and are
exercised by `test_live.py` with the same calls."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
from gg_binary import GG  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_the_offline_example_captures_checks_and_rejects():
    if not GG.exists():
        pytest.skip("gg binary not found (set GG_BIN)")
    env = dict(os.environ, GG_BIN=str(GG), PYTHONPATH=str(EXAMPLES.parent))
    out = subprocess.run([sys.executable, str(EXAMPLES / "check_offline.py")],
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert 'model "classifier" accepted by profile' in out.stdout
    # Since gg-ops/8 a sort is in the profile (ADR-0029): the example shows a one-hot encoding.
    assert "rejected as expected: operator 'aten.one_hot" in out.stdout
