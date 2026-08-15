#!/bin/bash
#SBATCH --job-name=abc
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --output=logs/abc_%j.out
#SBATCH --error=logs/abc_%j.err
#SBATCH --account=cvcs2026
#
# The reference comparison: A (no retrieval), B (retrieval) and C (agentic) for
# ONE model, on one job, so the three differ only in method — same weights, same
# endpoint, same prompt format, same examples. This is the table to start from
# and the one to re-run after changing anything in the pipeline.
#
# VARIANT names the run. Use it when trying a strategy, so results and the code
# that produced them stay paired: each predictions file gets a .meta.json with
# the commit, the branch and whether the tree was dirty.
#
#   sbatch scripts/run_abc.sh
#   VARIANT=lookup-first sbatch --export=ALL scripts/run_abc.sh
#   SMOKE=1 sbatch --export=ALL --time=00:40:00 scripts/run_abc.sh
#
# The model axis (C across sizes) is scripts/agentic/run_sweep.sh instead.

set -euo pipefail

MODEL="Qwen/Qwen3-VL-8B-Instruct"
TAG="qwen3vl8b"
GPU_UTIL=0.50      # the retriever and reranker share this GPU
MAX_LEN=32768
TOP_K=20
TOP_N=20
CONCURRENCY=8

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="/homes/$USER/vllm_venv"
OUT_DIR="outputs/abc/$TAG${VARIANT:+/$VARIANT}"

if [ "${SMOKE:-0}" = "1" ]; then
    LIMIT=(--limit 5); DEBUG=5; OUT_DIR="$OUT_DIR/smoke"
else
    LIMIT=(); DEBUG=3
fi

export HF_HOME="/work/cvcs2026/recursive_retrievers/hf_cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$HOME/.local/bin:$PATH"
export TFHUB_CACHE_DIR="/work/cvcs2026/recursive_retrievers/tfhub_cache"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/cvcs2026/recursive_retrievers/$USER/.uv_cache}"
mkdir -p "$UV_CACHE_DIR"
unset SSL_CERT_DIR

cd "$PROJECT_DIR"
mkdir -p logs "$OUT_DIR"
source scripts/lib/vllm.sh

ensure_vllm_venv
serve_model "$MODEL" "$GPU_UTIL" "$MAX_LEN"

echo "################ A — no retrieval"
uv run python src/vlm/run_inference.py \
    --model-name "$MODEL" --base-url "$BASE_URL" \
    --output "$OUT_DIR/predictions_A.jsonl" \
    --concurrency "$CONCURRENCY" --debug-samples "$DEBUG" "${LIMIT[@]}"

echo "################ B — retrieval (top-k=$TOP_K top-n=$TOP_N)"
uv run python src/vlm/run_inference.py \
    --model-name "$MODEL" --base-url "$BASE_URL" \
    --output "$OUT_DIR/predictions_B.jsonl" \
    --use-retrieval --top-k "$TOP_K" --rerank-top-n "$TOP_N" \
    --concurrency "$CONCURRENCY" --debug-samples "$DEBUG" "${LIMIT[@]}"

echo "################ C — agentic"
uv run python src/agent/run_inference.py \
    --model-name "$MODEL" --base-url "$BASE_URL" \
    --output "$OUT_DIR/predictions_C.jsonl" \
    --top-k "$TOP_K" --rerank-top-n "$TOP_N" \
    --concurrency 4 --debug-samples "$DEBUG" "${LIMIT[@]}"

stop_model

echo "################ scoring"
for s in A B C; do
    (cd "$PROJECT_DIR/evqa_eval" && uv run python score_evqa.py \
        --predictions "../$OUT_DIR/predictions_${s}.jsonl" \
        --output "../$OUT_DIR/results_${s}.json") || true
done

echo "################ summary  ($MODEL${VARIANT:+, variant $VARIANT})"
for s in A B C; do
    acc=$(python3 -c "import json;print(json.load(open('$OUT_DIR/results_${s}.json'))['accuracy_overall'])" 2>/dev/null || echo "n/a")
    echo "  $s: $acc"
done
echo "commit: $(git rev-parse --short HEAD) $(git status --porcelain >/dev/null && git diff --quiet || echo '(dirty)')"
