"""Load a trained model plus its tokenizer from a run directory or checkpoint."""

from __future__ import annotations

import json
import os

import torch

from .config import Config
from .device import pick_device
from .model import GPT
from .tokenizer import load as load_tokenizer


def load_model(path: str, device: str = "auto", tokenizer: str = ""):
    """path is a run dir (uses best.pt, else ckpt.pt) or a .pt file."""
    if os.path.isdir(path):
        best = os.path.join(path, "best.pt")
        path = best if os.path.exists(best) else os.path.join(path, "ckpt.pt")
    dev = pick_device(device)
    ck = torch.load(path, map_location=dev, weights_only=True)
    cfg = Config.from_dict(ck["config"])
    model = GPT(cfg.model).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    if not tokenizer:
        meta = os.path.join(cfg.train.data_dir, "meta.json")
        tokenizer = json.load(open(meta))["tokenizer"] if os.path.exists(meta) else "tokenizer/kyrie.json"
    return model, load_tokenizer(tokenizer), dev
