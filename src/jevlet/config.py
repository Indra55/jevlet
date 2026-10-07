"""Configuration, reproducibility, and run metadata."""

import copy
import importlib.metadata
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from . import ROOT

HEADS = ("choice", "score", "yesno")


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) else value
    return result


def load_config(path="config.yaml", smoke=False):
    with open(path) as handle:
        cfg = yaml.safe_load(handle)
    if smoke:
        cfg = merge(cfg, cfg["smoke"])
    cfg["is_smoke"] = smoke
    if not 2 <= cfg["options_min"] <= cfg["options_max"] <= 10:
        raise ValueError("Candidate counts must be between 2 and 10")
    if not 16 <= cfg["max_length"] <= 512:
        raise ValueError("max_length must be between 16 and 512")
    return cfg


def seed_all(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install a CUDA PyTorch build")
    return device


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_dir(run):
    if not run or Path(run).name != run or run in (".", ".."):
        raise ValueError("Run names must be a single directory name")
    return ROOT / "checkpoints" / run


def metadata(cfg, data=None):
    return {
        "config": cfg,
        "data": data,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "datasets", "scikit-learn", "numpy")},
        "cuda_available": torch.cuda.is_available(),
    }
