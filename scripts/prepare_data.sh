#!/usr/bin/env bash
# Build the full tokenizer and pretraining data. CPU only: run it on a cheap
# CPU droplet with a fast network, not on a GPU droplet.
#
#   source .venv/bin/activate
#   tmux new -s prep 'bash scripts/prepare_data.sh 2>&1 | tee prep.log'
#
# Settings can be overridden with environment variables, e.g.
#   TOKENS=2B bash scripts/prepare_data.sh          # a quicker first pass
#   SKIP_TOKENIZER=1 bash scripts/prepare_data.sh   # keep an existing tokenizer
set -euo pipefail

TOK_CHARS="${TOK_CHARS:-300M}"
TOKENS="${TOKENS:-10B}"
VAL_TOKENS="${VAL_TOKENS:-10M}"
TOKENIZER="${TOKENIZER:-tokenizer/kyrie.json}"
OUT="${OUT:-data/shards}"

cd "$(dirname "$0")/.."

if [ -z "${SKIP_TOKENIZER:-}" ] || [ ! -f "$TOKENIZER" ]; then
    if [ -f "$TOKENIZER" ]; then
        # Shards are only valid with the tokenizer that made them; never overwrite silently.
        echo "$TOKENIZER already exists. Move it away or set SKIP_TOKENIZER=1." >&2
        exit 1
    fi
    echo "==> tokenizer on $TOK_CHARS chars (code + Amharic + Afaan Oromo)"
    time python -m kyrie.tokenizer --out "$TOKENIZER" --chars "$TOK_CHARS"
fi

if [ -f "$OUT/meta.json" ]; then
    echo "$OUT/meta.json already exists. Move $OUT away to rebuild it." >&2
    exit 1
fi
echo "==> $TOKENS train tokens into $OUT"
time python -m kyrie.data --tokenizer "$TOKENIZER" --out "$OUT" --tokens "$TOKENS" --val_tokens "$VAL_TOKENS"

du -sh "$OUT"
