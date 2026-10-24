"""Generate text from a trained model.

    python -m kyrie.sample runs/tiny --prompt "def is_prime(n):"
"""

from __future__ import annotations

import argparse

import torch

from .checkpoint import load_model
from .tokenizer import EOT


def complete(model, tok, device, prompt: str, max_new_tokens=200, temperature=0.8,
             top_k=50, top_p=0.95) -> str:
    ids = tok.encode(prompt).ids or [tok.token_to_id(EOT)]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens, temperature, top_k, top_p,
                         stop_ids={tok.token_to_id(EOT)})
    return tok.decode(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", help="run directory or checkpoint .pt")
    p.add_argument("--prompt", default="def ")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--num", type=int, default=1)
    p.add_argument("--tokenizer", default="")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    model, tok, dev = load_model(args.run, args.device, args.tokenizer)
    for i in range(args.num):
        text = complete(model, tok, dev, args.prompt, args.max_new_tokens,
                        args.temperature, args.top_k, args.top_p)
        print(f"----- sample {i + 1} -----\n{args.prompt}{text}")


if __name__ == "__main__":
    main()
