"""Pick the best available device and precision: CUDA, Apple MPS, or CPU."""

from __future__ import annotations

import contextlib

import torch


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, requested: str = "auto") -> torch.dtype:
    if requested != "auto":
        return getattr(torch, requested)
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        # fp32 is the stable default on MPS. Try --dtype bfloat16 on macOS 14+ for speed.
        return torch.float32
    return torch.float32


def autocast(device: torch.device, dtype: torch.dtype):
    if dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
