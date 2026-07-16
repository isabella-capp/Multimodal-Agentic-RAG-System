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

VENV="/homes/$USER/vllm_venv"
VLLM_VERSION=0.25.1

echo "creating venv at $VENV (python 3.12) …"
uv venv "$VENV" --python 3.12

echo "installing vllm==$VLLM_VERSION …"
uv pip install --python "$VENV/bin/python" "vllm==$VLLM_VERSION"

echo "=== verify ==="
"$VENV/bin/python" -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, torch.version.cuda)"
"$VENV/bin/vllm" --version
echo "done"
