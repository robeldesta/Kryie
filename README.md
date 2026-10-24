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
python3 -m venv .venv && source .venv/bin/activate
# Intel Macs only: PyTorch 2.2.2 is the last build, on Python <= 3.12, and needs numpy < 2
# pip install "torch==2.2.2" "numpy<2"
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

## On a DigitalOcean GPU Droplet

```bash
pip install -e .
python -m kyrie.tokenizer --out tokenizer/kyrie.json --chars 1B
python -m kyrie.data --tokenizer tokenizer/kyrie.json --out data/shards --tokens 10B
python -m kyrie.train --preset base --data_dir data/shards              # 1x H100
torchrun --nproc_per_node=8 -m kyrie.train --preset base --grad_accum 1 # 8x H100
```

The `base` preset uses ~524k tokens per step, so 19k steps is ~10B tokens.
Reuse the tokenizer from your Mac rather than retraining it, so checkpoints stay
compatible.

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
