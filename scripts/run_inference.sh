#!/bin/bash
#SBATCH --job-name=qwen_inference
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/homes/%u/cvcs2026/logs/inference_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/inference_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail

PROJECT_DIR="/homes/$USER/cvcs2026"

export HF_HOME="/work/cvcs2026/recursive_retrievers/hf_cache/huggingface"
export HF_HUB_OFFLINE=1
export PATH="$HOME/.local/bin:$PATH"
unset SSL_CERT_DIR

cd "$PROJECT_DIR"
mkdir -p logs outputs

uv run python src/vlm/run_inference.py --output outputs/predictions.jsonl
