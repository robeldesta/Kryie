#!/usr/bin/env bash
# One-time setup on a fresh DigitalOcean droplet (Ubuntu 22.04/24.04).
# Works on CPU droplets (for data prep) and GPU droplets (for training).
#
#   git clone https://github.com/robeldesta/Kryie.git kyrie && cd kyrie
#   bash scripts/droplet_setup.sh [/mnt/kyrie_data]
#
# The optional argument is where data, tokenizer and runs live. Point it at a
# mounted DigitalOcean Volume so the data survives the droplet and can be
# re-attached to a GPU droplet later.
set -euo pipefail

DATA_ROOT="${1:-$HOME/kyrie_data}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> system packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-dev git tmux htop >/dev/null

echo "==> python env in $REPO_DIR/.venv"
python3 -m venv "$REPO_DIR/.venv"
# shellcheck disable=SC1091
source "$REPO_DIR/.venv/bin/activate"
pip install -q --upgrade pip
pip install -q -e "$REPO_DIR[dev]"

echo "==> data directories under $DATA_ROOT"
mkdir -p "$DATA_ROOT"/{tokenizer,data,runs,hf_cache}
for d in tokenizer data runs; do
    if [ ! -e "$REPO_DIR/$d" ]; then
        ln -s "$DATA_ROOT/$d" "$REPO_DIR/$d"
    fi
done
# Keep the Hugging Face cache on the data volume too, not the small boot disk.
if ! grep -q "HF_HOME=" "$HOME/.bashrc" 2>/dev/null; then
    echo "export HF_HOME=$DATA_ROOT/hf_cache" >> "$HOME/.bashrc"
fi
export HF_HOME="$DATA_ROOT/hf_cache"

echo "==> checks"
python - <<'EOF'
import os, torch
print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  gpu {i}: {torch.cuda.get_device_name(i)}")
print(f"cpus: {os.cpu_count()}")
EOF
pytest -q "$REPO_DIR/tests"

df -h "$DATA_ROOT" | tail -1
echo
echo "Done. Next: source .venv/bin/activate && bash scripts/prepare_data.sh"
