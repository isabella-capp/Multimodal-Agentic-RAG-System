#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#SBATCH --account=cvcs2026

# Cheap check that a remote model works with our agent before committing to the
# full sweep — above all, that its provider accepts tool_choice="required"
# (the forced first call). A handful of examples, full traces in the log.
#
#   export LLM_API_KEY=sk-or-v1-...
#   sbatch --export=ALL scripts/run_smoke.sh
#
# Then read logs/smoke_<jobid>.err: if every example carries an error, the
# provider rejects the forced call — add --no-force-first-tool to the remote
# model in run_model_sweep.sh.

set -euo pipefail

MODEL="qwen/qwen2.5-vl-72b-instruct"
TAG="qwen72b"
LIMIT=5

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PRED="outputs/agentic/smoke_${TAG}.jsonl"

export HF_HOME="/work/cvcs2026/recursive_retrievers/hf_cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
unset SSL_CERT_DIR

cd "$PROJECT_DIR"
mkdir -p logs outputs/agentic

: "${LLM_API_KEY:?LLM_API_KEY is not set — export it and submit with 'sbatch --export=ALL'}"

rm -f "$PRED" "${PRED%.jsonl}.metrics.json"  # the eval resumes, so start clean

uv run python src/agent/run_agentic_eval.py \
    --model-name "$MODEL" \
    --base-url "https://openrouter.ai/api/v1" \
    --output "$PRED" \
    --limit "$LIMIT" --concurrency 1 --debug-samples "$LIMIT"

echo "################ metrics"
cat "${PRED%.jsonl}.metrics.json"
