#!/bin/bash
#SBATCH --job-name=recall_ablation
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_RTX_A5000_24G|gpu_RTX6000_24G
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/homes/%u/cvcs2026/logs/recall_ablation_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/recall_ablation_%j.err
#SBATCH --account=cvcs2026
#
# Full paragraph-level retrieval ablation study.
#
# Evaluates 17 configurations in a SINGLE job so that the expensive model
# loading (EVA-CLIP, FAISS, BGE cross-encoder) happens only once.
#
# Configurations
# ==============
#   A. BM25 only:        bm25_top5 / top10 / top20 / top50
#   B. BGE only:         bge_top5  / top10 / top20
#   C. BM25 → BGE:       bm25_20_bge_5/10/20 | bm25_50_bge_5/10/20
#   D. BM25 + BGE (RRF): rrf_top5  / top10 / top20
#
# For C modes, the script also records the intermediate BM25 rank so we
# can distinguish "BM25 failed" from "BGE re-ordered away" in the report.
#
# Usage
# =====
#   # Full ablation (all 17 modes, 1000 examples):
#   sbatch scripts/retrieval/run_ablation_paragraphs.sh
#
#   # Smoke test (50 examples for a quick sanity check):
#   SMOKE=1 sbatch scripts/retrieval/run_ablation_paragraphs.sh
#
#   # Custom subset:
#   MODES="bm25_top20 rrf_top5 rrf_top10 rrf_top20" sbatch ...
#
#   # Re-print table from existing output (no GPU needed):
#   REPORT_ONLY=1 sbatch --partition=all_usr_prod --gres="" ...

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
TOP_K="${TOP_K:-20}"            # EVA-CLIP FAISS neighbours
RRF_K="${RRF_K:-60}"           # RRF smoothing constant
CROSS_ENCODER="${CROSS_ENCODER_MODEL:-BAAI/bge-reranker-base}"

# Default: all 17 new-style modes in one pass
MODES="${MODES:-bm25_top5 bm25_top10 bm25_top20 bm25_top50 \
               bge_top5 bge_top10 bge_top20 \
               bm25_20_bge_5 bm25_20_bge_10 bm25_20_bge_20 \
               bm25_50_bge_5 bm25_50_bge_10 bm25_50_bge_20 \
               rrf_top5 rrf_top10 rrf_top20}"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CODE_DIR="${CODE_DIR:-$PROJECT_DIR}"
OUT_DIR="outputs/retrieval/ablation"

if [ "${SMOKE:-0}" = "1" ]; then
    LIMIT=(--limit 50)
    OUT_DIR="$OUT_DIR/smoke"
    echo "[SMOKE] limiting to 50 examples"
else
    LIMIT=()
fi

REPORT_ONLY_FLAG=()
if [ "${REPORT_ONLY:-0}" = "1" ]; then
    REPORT_ONLY_FLAG=(--report-only)
fi

export HF_HOME="/work/cvcs2026/recursive_retrievers/hf_cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export CROSS_ENCODER_MODEL="$CROSS_ENCODER"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/cvcs2026/recursive_retrievers/$USER/.uv_cache}"
mkdir -p "$UV_CACHE_DIR"
unset SSL_CERT_DIR

cd "$PROJECT_DIR"
mkdir -p "${LOG_DIR:-logs}" "$OUT_DIR"

OUTPUT="$OUT_DIR/ablation_topk${TOP_K}_rrf${RRF_K}.jsonl"

echo "================================================================"
echo "Paragraph Retrieval Ablation Study"
echo "  top_k (FAISS articles) : $TOP_K"
echo "  rrf_k                  : $RRF_K"
echo "  cross-encoder          : $CROSS_ENCODER"
echo "  output                 : $OUTPUT"
echo "  modes                  : $MODES"
echo "================================================================"

# shellcheck disable=SC2086
uv run python "$CODE_DIR"/src/retrieval/experiments/compute_recall_paragraphs.py \
    --top-k "$TOP_K" \
    --rrf-k "$RRF_K" \
    --cross-encoder-model "$CROSS_ENCODER" \
    --output "$OUTPUT" \
    --modes $MODES \
    "${LIMIT[@]}" \
    "${REPORT_ONLY_FLAG[@]}"

echo "================================================================"
echo "Done. Results in $OUTPUT"
echo "================================================================"

# Convenience: re-print a clean per-type breakdown
echo ""
echo "--- BM25 only ---"
# shellcheck disable=SC2086
uv run python "$CODE_DIR"/src/retrieval/experiments/compute_recall_paragraphs.py \
    --report-only --output "$OUTPUT" \
    --modes bm25_top5 bm25_top10 bm25_top20 bm25_top50 2>/dev/null || true

echo "--- BGE only ---"
uv run python "$CODE_DIR"/src/retrieval/experiments/compute_recall_paragraphs.py \
    --report-only --output "$OUTPUT" \
    --modes bge_top5 bge_top10 bge_top20 2>/dev/null || true

echo "--- BM25 -> BGE ---"
# shellcheck disable=SC2086
uv run python "$CODE_DIR"/src/retrieval/experiments/compute_recall_paragraphs.py \
    --report-only --output "$OUTPUT" \
    --modes bm25_20_bge_5 bm25_20_bge_10 bm25_20_bge_20 \
            bm25_50_bge_5 bm25_50_bge_10 bm25_50_bge_20 2>/dev/null || true

echo "--- RRF ---"
uv run python "$CODE_DIR"/src/retrieval/experiments/compute_recall_paragraphs.py \
    --report-only --output "$OUTPUT" \
    --modes rrf_top5 rrf_top10 rrf_top20 2>/dev/null || true
