"""Train and load Kyrie's byte-level BPE tokenizer.

Byte-level means any input (code, Ge'ez script, emoji) can be encoded with no
unknown tokens. Digits are split one per token, which helps with arithmetic.

    python -m kyrie.tokenizer --out tokenizer/kyrie.json --chars 200_000_000
"""

from __future__ import annotations

import argparse
import itertools

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from .sources import DEFAULT_TOKENIZER_MIX, iter_mix, parse_count

EOT = "<|endoftext|>"
SPECIAL_TOKENS = [
    EOT,
    "<|pad|>",
    # fill-in-the-middle, for editor-style completion later
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    # chat turns, for instruction tuning later
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"\d"), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def train(sources: list[str], vocab_size: int, max_chars: int, out: str, seed: int = 0) -> Tokenizer:
    def texts():
        budget = max_chars
        for doc in iter_mix(sources, seed=seed):
            doc = doc[:20_000]  # long files add little beyond their first part
            budget -= len(doc)
            yield doc
            if budget <= 0:
                return

    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True,
    )
    tok.train_from_iterator(texts(), trainer=trainer)
    tok.save(out)
    return tok


def load(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def report(tok: Tokenizer) -> None:
    """Print chars-per-token on a few samples, a quick sanity check of coverage."""
    samples = {
        "python": "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
        "amharic": "ኢትዮጵያ በምሥራቅ አፍሪካ የምትገኝ ሀገር ናት። ዋና ከተማዋ አዲስ አበባ ነው።",
        "oromo": "Itoophiyaan biyya Afrikaa Bahaa keessatti argamtu dha. Magaalaan guddoon ishee Finfinnee dha.",
        "english": "The quick brown fox jumps over the lazy dog.",
    }
    for name, text in samples.items():
        n = len(tok.encode(text).ids)
        print(f"  {name:8s} {len(text):4d} chars -> {n:4d} tokens ({len(text) / n:.2f} chars/token)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", action="append", help="weight@spec, repeatable (see kyrie/sources.py)")
    p.add_argument("--vocab_size", type=int, default=32768)
    p.add_argument("--chars", type=parse_count, default=parse_count("200M"))
    p.add_argument("--out", default="tokenizer/kyrie.json")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok = train(args.source or DEFAULT_TOKENIZER_MIX, args.vocab_size, args.chars, args.out, args.seed)
    print(f"saved {args.out} (vocab {tok.get_vocab_size()})")
    report(tok)


if __name__ == "__main__":
    main()
