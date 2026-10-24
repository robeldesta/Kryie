"""Model and training configuration, with named presets.

Presets are a starting point; any field can be overridden from the command line,
e.g. `python -m kyrie.train --preset tiny --max_steps 200 --lr 1e-3`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    seq_len: int = 1024
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 12  # < n_heads gives grouped-query attention
    ffn_mult: float = 8 / 3  # SwiGLU hidden size = ffn_mult * d_model, rounded to 64
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True

    @property
    def ffn_dim(self) -> int:
        return int(round(self.ffn_mult * self.d_model / 64) * 64)


@dataclass
class TrainConfig:
    # data
    data_dir: str = "data/shards"
    out_dir: str = "runs/default"
    # optimisation
    batch_size: int = 16  # sequences per micro-batch
    grad_accum: int = 4  # micro-batches per optimizer step
    max_steps: int = 5000
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # bookkeeping
    eval_every: int = 250
    eval_batches: int = 20
    save_every: int = 1000
    log_every: int = 10
    seed: int = 1337
    # hardware
    device: str = "auto"  # auto | cuda | mps | cpu
    dtype: str = "auto"  # auto | bfloat16 | float16 | float32
    compile: bool = False
    resume: bool = True
    wandb_project: str = ""  # set to log to Weights & Biases


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(model=ModelConfig(**d["model"]), train=TrainConfig(**d["train"]))


def _preset(model: dict, train: dict) -> Config:
    return Config(model=ModelConfig(**model), train=TrainConfig(**train))


PRESETS: dict[str, Config] = {
    # A few minutes on a laptop. Used for smoke tests and debugging.
    "tiny": _preset(
        dict(seq_len=256, d_model=128, n_layers=4, n_heads=4, n_kv_heads=4),
        dict(batch_size=16, grad_accum=1, max_steps=300, lr=1e-3, warmup_steps=30,
             eval_every=100, eval_batches=5, save_every=300, out_dir="runs/tiny"),
    ),
    # ~30M params. Trains meaningfully on an Apple Silicon Mac overnight.
    "small": _preset(
        dict(seq_len=512, d_model=384, n_layers=8, n_heads=6, n_kv_heads=6),
        dict(batch_size=16, grad_accum=4, max_steps=20000, lr=8e-4, warmup_steps=500,
             out_dir="runs/small"),
    ),
    # ~100M params (75M non-embedding). The main target, sized for a single H100.
    # 32 seq * 2048 tokens * 8 accum = ~524k tokens/step; 19k steps = ~10B tokens.
    "base": _preset(
        dict(seq_len=2048, d_model=768, n_layers=12, n_heads=12, n_kv_heads=4),
        dict(batch_size=32, grad_accum=8, max_steps=19000, lr=6e-4, warmup_steps=1000,
             eval_every=500, save_every=1000, compile=True, out_dir="runs/base"),
    ),
}


def parse_config(argv: list[str] | None = None) -> Config:
    """Build a Config from --preset plus per-field overrides."""
    p = argparse.ArgumentParser(description="Kyrie training")
    p.add_argument("--preset", default="tiny", choices=sorted(PRESETS))
    p.add_argument("--config", default="", help="JSON config file (overrides preset)")
    for section in (ModelConfig, TrainConfig):
        for f in dataclasses.fields(section):
            kind = type(f.default)
            if kind is bool:
                p.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"))
            else:
                p.add_argument(f"--{f.name}", type=kind)
    args = p.parse_args(argv)

    cfg = PRESETS[args.preset]
    cfg = Config.from_dict(json.loads(json.dumps(cfg.to_dict())))  # deep copy
    if args.config:
        with open(args.config) as fh:
            cfg = Config.from_dict(json.load(fh))
    for section in (cfg.model, cfg.train):
        for f in dataclasses.fields(section):
            v = getattr(args, f.name)
            if v is not None:
                setattr(section, f.name, v)
    return cfg
