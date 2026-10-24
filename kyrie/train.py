"""Pretrain Kyrie.

    python -m kyrie.train --preset tiny                      # laptop smoke test
    python -m kyrie.train --preset small --data_dir data/shards
    torchrun --nproc_per_node=8 -m kyrie.train --preset base # multi-GPU on one node

Checkpoints go to <out_dir>/ckpt.pt and runs resume from there automatically.
"""

from __future__ import annotations

import json
import math
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import Config, parse_config
from .data import Batches
from .device import autocast, pick_device, pick_dtype, synchronize
from .model import GPT


def lr_at(step: int, cfg) -> float:
    """Linear warmup, then cosine decay to min_lr_ratio * lr."""
    t = cfg.train
    if step < t.warmup_steps:
        return t.lr * (step + 1) / t.warmup_steps
    progress = min(1.0, (step - t.warmup_steps) / max(1, t.max_steps - t.warmup_steps))
    min_lr = t.lr * t.min_lr_ratio
    return min_lr + 0.5 * (t.lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, batches: Batches, n: int, ctx) -> float:
    model.eval()
    losses = []
    for _ in range(n):
        x, y = batches.next()
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save(path, model, opt, step, cfg, best_val):
    raw = model.module if isinstance(model, DDP) else model
    raw = getattr(raw, "_orig_mod", raw)
    tmp = path + ".tmp"
    torch.save({"model": raw.state_dict(), "optimizer": opt.state_dict(), "step": step,
                "config": cfg.to_dict(), "best_val": best_val}, tmp)
    os.replace(tmp, path)


def main(argv=None):
    cfg: Config = parse_config(argv)
    t = cfg.train

    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank, world = 0, 1
        device = pick_device(t.device)
    master = rank == 0
    dtype = pick_dtype(device, t.dtype)
    ctx = autocast(device, dtype)
    torch.manual_seed(t.seed + rank)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Vocab size comes from the data so model and shards always agree.
    meta_path = os.path.join(t.data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            cfg.model.vocab_size = json.load(fh)["vocab_size"]

    os.makedirs(t.out_dir, exist_ok=True)
    ckpt_path = os.path.join(t.out_dir, "ckpt.pt")
    model = GPT(cfg.model).to(device)
    opt = model.configure_optimizer(t.weight_decay, t.lr, (t.beta1, t.beta2), device.type)
    step, best_val = 0, float("inf")
    if t.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, best_val = ck["step"], ck.get("best_val", best_val)
        if master:
            print(f"resumed from {ckpt_path} at step {step}")

    use_scaler = dtype == torch.float16
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)
    else:  # torch < 2.3, e.g. the last builds for Intel Macs
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    if t.compile and device.type == "cuda":
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[device.index])

    train_b = Batches(t.data_dir, "train", cfg.model.seq_len, t.batch_size, device, t.seed + step, rank)
    val_b = Batches(t.data_dir, "val", cfg.model.seq_len, t.batch_size, device, t.seed, 0)

    tokens_per_step = t.batch_size * t.grad_accum * cfg.model.seq_len * world
    if master:
        raw = model.module if ddp else model
        n_params = getattr(raw, "_orig_mod", raw).num_params()
        print(f"device={device} dtype={dtype} world={world}")
        print(f"params (non-embedding) {n_params / 1e6:.1f}M, "
              f"{tokens_per_step:,} tokens/step, {t.max_steps} steps "
              f"= {tokens_per_step * t.max_steps / 1e9:.2f}B tokens "
              f"(train split has {train_b.tokens / 1e9:.2f}B)")
        with open(os.path.join(t.out_dir, "config.json"), "w") as fh:
            json.dump(cfg.to_dict(), fh, indent=2)
        log = open(os.path.join(t.out_dir, "log.jsonl"), "a")
        wandb = None
        if t.wandb_project:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(project=t.wandb_project, config=cfg.to_dict(), resume="allow",
                       name=os.path.basename(t.out_dir))

    model.train()
    t0 = time.time()
    while step < t.max_steps:
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        loss_acc = 0.0
        for micro in range(t.grad_accum):
            x, y = train_b.next()
            if ddp:
                model.require_backward_grad_sync = micro == t.grad_accum - 1
            with ctx:
                _, loss = model(x, y)
                loss = loss / t.grad_accum
            scaler.scale(loss).backward()
            loss_acc += loss.item()

        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1

        if master and step % t.log_every == 0:
            synchronize(device)
            dt = time.time() - t0
            t0 = time.time()
            tok_s = tokens_per_step * t.log_every / dt
            rec = {"step": step, "loss": round(loss_acc, 4), "lr": lr,
                   "gnorm": round(float(gnorm), 3), "tok_per_s": int(tok_s)}
            print(f"step {step:6d} | loss {loss_acc:.4f} | lr {lr:.2e} | "
                  f"gnorm {float(gnorm):.2f} | {tok_s:,.0f} tok/s")
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if wandb:
                wandb.log(rec, step=step)

        last = step == t.max_steps
        if master and (step % t.eval_every == 0 or last):
            val = evaluate(model, val_b, t.eval_batches, ctx)
            print(f"step {step:6d} | val loss {val:.4f} | val ppl {math.exp(val):.2f}")
            log.write(json.dumps({"step": step, "val_loss": round(val, 4)}) + "\n")
            log.flush()
            if wandb:
                wandb.log({"val_loss": val}, step=step)
            if val < best_val:
                best_val = val
                save(os.path.join(t.out_dir, "best.pt"), model, opt, step, cfg, best_val)
        if master and (step % t.save_every == 0 or last):
            save(ckpt_path, model, opt, step, cfg, best_val)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
