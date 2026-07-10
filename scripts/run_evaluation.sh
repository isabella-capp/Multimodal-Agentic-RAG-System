#!/bin/bash
#SBATCH --job-name=evaluate_model
#SBATCH --partition=all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/homes/%u/cvcs2026/logs/evaluate_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/evaluate_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail

PROJECT_DIR="/homes/$USER/cvcs2026"

export TFHUB_CACHE_DIR="/work/cvcs2026/recursive_retrievers/tfhub_cache"
export PATH="$HOME/.local/bin:$PATH"
unset SSL_CERT_DIR

mkdir -p "$PROJECT_DIR/logs" "$PROJECT_DIR/outputs"

# BEM runs on CPU; the eval lives in a separate uv project (its own .venv).
cd "$PROJECT_DIR/evqa_eval"
uv run python score_evqa.py \
    --predictions ../outputs/predictions_rag.jsonl \
    --output ../outputs/results_rag.json
