#!/bin/bash
#SBATCH --job-name=vllm_venv_setup
#SBATCH --partition=all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/homes/%u/cvcs2026/logs/vllm_venv_setup_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/vllm_venv_setup_%j.err
#SBATCH --account=cvcs2026


set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
unset SSL_CERT_DIR

export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/cvcs2026/recursive_retrievers/$USER/.uv_cache}"
mkdir -p "$UV_CACHE_DIR"

VENV="/homes/$USER/vllm_venv"
VLLM_VERSION=0.25.1

echo "creating venv at $VENV (python 3.12) …"
uv venv "$VENV" --python 3.12 --clear

echo "installing vllm==$VLLM_VERSION …"
uv pip install --python "$VENV/bin/python" "vllm==$VLLM_VERSION"

echo "=== verify ==="
"$VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"
test -x "$VENV/bin/vllm" && echo "vllm binary present at $VENV/bin/vllm"
echo "done"
