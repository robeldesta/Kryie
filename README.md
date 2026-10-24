# Kyrie

A small code language model trained from scratch in plain PyTorch. Code first;
Amharic and Afaan Oromo come later, but the tokenizer already covers them.

The same code runs on an Apple Silicon Mac (MPS), on CPU, and on NVIDIA GPUs
(single GPU or multi-GPU with `torchrun`).

## Layout

| File | What it does |
| --- | --- |
| `kyrie/model.py` | Decoder-only transformer: RMSNorm, RoPE, SwiGLU, grouped-query attention, KV-cache sampling |
| `kyrie/config.py` | Model and training settings, with `tiny`, `small` and `base` presets |
| `kyrie/sources.py` | Where text comes from (Hugging Face streams, local folders), code filters, dedup |
| `kyrie/tokenizer.py` | Trains a 32k byte-level BPE tokenizer on code plus Amharic and Oromo |
| `kyrie/data.py` | Tokenizes a corpus into uint16 shards and samples training batches from them |
| `kyrie/train.py` | Training loop: grad accumulation, cosine LR, mixed precision, checkpoints, resume, DDP |
| `kyrie/sample.py` | Generate from a checkpoint |
| `kyrie/eval/humaneval.py` | HumanEval pass@k, running generated code in a subprocess with a timeout |

## Presets

| Preset | Params (non-embedding) | Context | Where |
| --- | --- | --- | --- |
| `tiny` | ~0.8M | 256 | Smoke test, a few minutes anywhere |
| `small` | ~14M | 512 | An Apple Silicon Mac overnight (too slow on Intel CPU) |
| `base` | ~75M (~100M with embeddings) | 2048 | One H100, ~10B tokens |

## Quickstart on a Mac

```bash
python3 -m venv .venv && source .venv/bin/activate   # or: conda create -n kyrie python=3.11
# Intel Macs only: PyTorch 2.2.2 is the last build (Python <= 3.12) and needs numpy 1.x,
# so pin the libraries that would otherwise pull in numpy 2:
# pip install "torch==2.2.2" "numpy<2" "pyarrow<17" "datasets<3"
pip install -e ".[dev]"
pytest -q                                   # unit tests, CPU only, ~30s

# 1. Tokenizer: streams ~200M chars of code + Amharic + Oromo from Hugging Face
python -m kyrie.tokenizer --out tokenizer/kyrie.json --chars 200M

# 2. Data: 100M tokens of filtered, deduplicated Python
python -m kyrie.data --tokenizer tokenizer/kyrie.json --out data/shards --tokens 100M

# 3. Train. tiny first to check everything works, then small.
python -m kyrie.train --preset tiny  --data_dir data/shards
python -m kyrie.train --preset small --data_dir data/shards

# 4. Try it
python -m kyrie.sample runs/small --prompt "def is_prime(n):" --num 3
python -m kyrie.eval.humaneval runs/small --limit 20
```

Fully offline, any folder of `.py` files works as a source, e.g.
`--source "dir:/path/to/some/python/projects"`.

Override any setting on the command line: `--batch_size 8 --max_steps 2000 --lr 1e-3`.
Training resumes from `runs/<name>/ckpt.pt` automatically; pass `--resume false` to start over.

## On DigitalOcean

Data prep needs a fast network and CPUs but no GPU, so do it on a cheap CPU
droplet and only rent a GPU droplet for training. Keep everything on a
DigitalOcean **Volume** (in the same region as your GPU droplets) so the data
moves between droplets by detaching and re-attaching the volume.

**1. Data prep on a CPU droplet** (8+ vCPUs, 16+ GB RAM; a 100 GB volume holds 10B tokens with room to spare)

```bash
git clone https://github.com/robeldesta/Kryie.git kyrie && cd kyrie
bash scripts/droplet_setup.sh /mnt/<your_volume>
source .venv/bin/activate
tmux new -s prep 'bash scripts/prepare_data.sh 2>&1 | tee prep.log'   # 300M-char tokenizer + 10B tokens
```

`TOKENS=2B bash scripts/prepare_data.sh` makes a quicker first pass. If the
source runs out before the target, `meta.json` records how many tokens it really
got; training just samples the data more than once.

**2. Training on a GPU droplet.** Power off the CPU droplet, detach the volume, create the GPU droplet in the same region, attach the volume, then:

```bash
git clone https://github.com/robeldesta/Kryie.git kyrie && cd kyrie
bash scripts/droplet_setup.sh /mnt/<your_volume>
source .venv/bin/activate
python -m kyrie.train --preset base --max_steps 200 --out_dir runs/base-test   # short check: speed and loss
tmux new -s train 'python -m kyrie.train --preset base 2>&1 | tee runs/base.log'   # 1x H100
torchrun --nproc_per_node=8 -m kyrie.train --preset base --grad_accum 1             # 8x H100
```

The `base` preset uses ~524k tokens per step, so 19k steps is ~10B tokens.
Checkpoints are saved every 1000 steps and runs resume automatically, so a
droplet can be destroyed and recreated without losing more than that. Destroy
GPU droplets as soon as a run finishes: they bill while powered off.

## Data sources

Defaults (change with `--source "weight@spec"`, repeatable):

- Code: `codeparrot/codeparrot-clean` (Python from GitHub). For more languages and
  more tokens, `bigcode/the-stack-v2-dedup` or `bigcode/starcoderdata` (gated: accept
  the terms on Hugging Face and run `huggingface-cli login`).
- Tokenizer only, for now: Wikipedia in Amharic (`20231101.am`) and Afaan Oromo (`20231101.om`).

## Roadmap

1. Codebase (this). ✓
2. Data at scale: Stack v2 subset, near-dedup, decontaminate against HumanEval/MBPP.
3. Pretrain `base` on ~10B tokens; track HumanEval/MBPP.
4. Continue pretraining with Amharic and Afaan Oromo, then a light instruction tune.
