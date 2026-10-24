"""Tokenize a corpus into flat uint16 shards, and sample training batches from them.

    python -m kyrie.data --tokenizer tokenizer/kyrie.json --out data/shards --tokens 100M

Each shard is a raw array of token ids with <|endoftext|> between documents.
Training samples random windows across all shards via np.memmap, so nothing
has to fit in RAM.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from .sources import DEFAULT_CODE, ExactDedup, iter_mix, keep_code, parse_count
from .tokenizer import EOT, load

SHARD_TOKENS = 100_000_000


class ShardWriter:
    def __init__(self, out_dir: str, prefix: str, shard_tokens: int = SHARD_TOKENS):
        self.out_dir, self.prefix, self.shard_tokens = out_dir, prefix, shard_tokens
        self.buf = np.empty(shard_tokens, dtype=np.uint16)
        self.n = 0
        self.shard = 0
        self.total = 0

    def write(self, ids: list[int]) -> None:
        arr = np.asarray(ids, dtype=np.uint16)
        while len(arr):
            take = min(len(arr), self.shard_tokens - self.n)
            self.buf[self.n : self.n + take] = arr[:take]
            self.n += take
            self.total += take
            arr = arr[take:]
            if self.n == self.shard_tokens:
                self.flush()

    def flush(self) -> None:
        if self.n:
            path = os.path.join(self.out_dir, f"{self.prefix}_{self.shard:04d}.bin")
            self.buf[: self.n].tofile(path)
            self.shard += 1
            self.n = 0


def prepare(sources, tokenizer_path, out_dir, max_tokens, val_tokens, filter_code=True, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    tok = load(tokenizer_path)
    assert tok.get_vocab_size() <= 65536, "uint16 shards need vocab <= 65536"
    eot = tok.token_to_id(EOT)
    dedup = ExactDedup()
    val = ShardWriter(out_dir, "val")
    train = ShardWriter(out_dir, "train")
    stats = {"docs_seen": 0, "docs_kept": 0}

    batch: list[str] = []

    def flush_batch():
        for enc in tok.encode_batch(batch):
            ids = enc.ids + [eot]
            (val if val.total < val_tokens else train).write(ids)
        batch.clear()

    for doc in iter_mix(sources, seed=seed):
        stats["docs_seen"] += 1
        if filter_code and not keep_code(doc):
            continue
        if not dedup(doc):
            continue
        stats["docs_kept"] += 1
        batch.append(doc)
        if len(batch) >= 512:
            flush_batch()
            if stats["docs_kept"] % 20_000 < 512:
                print(f"  {train.total / 1e6:8.1f}M train tokens, {stats['docs_kept']:,} docs")
            if train.total >= max_tokens:
                break
    if batch:
        flush_batch()
    val.flush()
    train.flush()

    meta = {
        "tokenizer": os.path.abspath(tokenizer_path),
        "vocab_size": tok.get_vocab_size(),
        "eot_id": eot,
        "train_tokens": train.total,
        "val_tokens": val.total,
        "sources": sources,
        **stats,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))
    return meta


class Batches:
    """Random (x, y) windows of length seq_len drawn across a split's shards."""

    def __init__(self, data_dir: str, split: str, seq_len: int, batch_size: int,
                 device: torch.device, seed: int = 0, rank: int = 0):
        paths = sorted(glob.glob(os.path.join(data_dir, f"{split}_*.bin")))
        if not paths:
            raise FileNotFoundError(f"no {split}_*.bin shards in {data_dir}; run `python -m kyrie.data` first")
        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        self.shards = [s for s in self.shards if len(s) > seq_len + 1]
        sizes = np.array([len(s) for s in self.shards], dtype=np.float64)
        self.p = sizes / sizes.sum()
        self.tokens = int(sizes.sum())
        self.seq_len, self.batch_size, self.device = seq_len, batch_size, device
        self.rng = np.random.default_rng(seed + 1000 * rank)

    def next(self):
        T = self.seq_len
        which = self.rng.choice(len(self.shards), size=self.batch_size, p=self.p)
        rows = []
        for s in which:
            shard = self.shards[s]
            i = self.rng.integers(0, len(shard) - T - 1)
            rows.append(np.asarray(shard[i : i + T + 1], dtype=np.int64))
        buf = torch.from_numpy(np.stack(rows))
        x, y = buf[:, :-1], buf[:, 1:]
        if self.device.type == "cuda":
            return (x.pin_memory().to(self.device, non_blocking=True),
                    y.pin_memory().to(self.device, non_blocking=True))
        return x.to(self.device), y.to(self.device)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", help="weight@spec, repeatable (see kyrie/sources.py)")
    p.add_argument("--tokenizer", default="tokenizer/kyrie.json")
    p.add_argument("--out", default="data/shards")
    p.add_argument("--tokens", type=parse_count, default=parse_count("100M"))
    p.add_argument("--val_tokens", type=parse_count, default=parse_count("2M"))
    p.add_argument("--no_code_filter", action="store_true", help="skip code quality filter (for prose)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    prepare(args.source or DEFAULT_CODE, args.tokenizer, args.out, args.tokens,
            args.val_tokens, filter_code=not args.no_code_filter, seed=args.seed)


if __name__ == "__main__":
    main()
