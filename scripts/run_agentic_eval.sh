#!/bin/bash
#SBATCH --job-name=agentic_rag
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/homes/%u/cvcs2026/logs/agentic_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/agentic_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail

PROJECT_DIR="/homes/$USER/cvcs2026"

export HF_HOME="/work/cvcs2026/recursive_retrievers/hf_cache/huggingface"
export HF_HUB_OFFLINE=1
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
unset SSL_CERT_DIR

cd "$PROJECT_DIR"
mkdir -p logs outputs

# ── Agentic RAG inference ────────────────────────────────────────────
# Adjust --top-k and --rerank-top-n to the best values from ablation.
# Adjust --max-iterations to control how many times the agent can
# reformulate its query (3 is a good starting point).

uv run python src/agent/run_agentic_eval.py \
    --output outputs/predictions_agentic.jsonl \
    --top-k 20 \
    --rerank-top-n 5 \
    --max-iterations 3 \
    --debug-samples 5
