"""Local typed decisions. Downloaded assets and caches stay in the workspace."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")

__version__ = "0.1.0"


def __getattr__(name):
    if name == "Jevlet":
        from .model import Jevlet
        return Jevlet
    raise AttributeError(name)

