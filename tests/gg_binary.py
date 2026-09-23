"""The `gg` binary that the tests use as the reference checker.

`GG_BIN` names it explicitly; otherwise `gg` is looked up on `PATH`. Tests that need it are skipped
when it cannot be found, and say so.
"""

import os
import shutil
from pathlib import Path


def locate() -> Path:
    explicit = os.environ.get("GG_BIN")
    if explicit:
        return Path(explicit)
    found = shutil.which("gg")
    return Path(found) if found else Path("gg-not-found")


GG = locate()
