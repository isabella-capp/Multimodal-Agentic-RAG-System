#!/bin/bash
#SBATCH --job-name=lookup_merge_ablation
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --output=logs/lookup_merge_ablation_%j.out
#SBATCH --error=logs/lookup_merge_ablation_%j.err
#SBATCH --account=cvcs2026
#
# Ablation: category-priority merge vs RRF merge in lookup_articles.
#
# Runs two evaluation modes in a single job:
#
#   gt_title   (ceiling / perfect namer)
#       Uses ground-truth Wikipedia titles as lookup queries.
#       Always runs; backward-compatible with previous runs.
#
#   predicted  (realistic / production)
#       Uses VLM-predicted names from a naming_probe JSONL.
#       Only runs when NAMES_FILE is set (see below).
#
# CPU only: pure SQLite, no model loading, no GPU required.
#
# Usage
# =====
#   # Standard run -- ceiling only (backward-compatible):
#   scripts/submit.sh scripts/retrieval/run_lookup_merge_ablation.sh
#
#   # Both modes -- ceiling + realistic VLM names:
#   NAMES_FILE=outputs/agentic/naming_probe.jsonl \
#     scripts/submit.sh scripts/retrieval/run_lookup_merge_ablation.sh
#
#   # Choose a different naming variant (default: full):
#   NAMES_FILE=outputs/agentic/naming_probe.jsonl NAMES_VARIANT=center80 \
#     scripts/submit.sh scripts/retrieval/run_lookup_merge_ablation.sh
#
#   # Smoke test (50 examples):
#   SMOKE=1 scripts/submit.sh scripts/retrieval/run_lookup_merge_ablation.sh

set -euo pipefail

# -- Configuration -------------------------------------------------------------
LIMIT="${LIMIT:-1000}"
LOOKUP_LIMIT="${LOOKUP_LIMIT:-5}"
RRF_K="${RRF_K:-60}"
NAMES_FILE="${NAMES_FILE:-}"          # empty = gt_title mode only
NAMES_VARIANT="${NAMES_VARIANT:-full}"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CODE_DIR="${CODE_DIR:-$PROJECT_DIR}"
OUT_DIR="outputs/retrieval"

if [ "${SMOKE:-0}" = "1" ]; then
    LIMIT=50
    OUT_DIR="$OUT_DIR/smoke"
    echo "[SMOKE] limiting to 50 examples"
fi

export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/cvcs2026/recursive_retrievers/$USER/.uv_cache}"
mkdir -p "$UV_CACHE_DIR"
unset SSL_CERT_DIR

cd "$PROJECT_DIR"
mkdir -p "${LOG_DIR:-logs}" "$OUT_DIR"

# Output filename encodes key parameters
if [ -n "$NAMES_FILE" ]; then
    OUTPUT="$OUT_DIR/lookup_merge_ablation_both_lim${LOOKUP_LIMIT}_rrf${RRF_K}.json"
else
    OUTPUT="$OUT_DIR/lookup_merge_ablation_lim${LOOKUP_LIMIT}_rrf${RRF_K}.json"
fi

echo "================================================================"
echo "lookup_articles merge strategy ablation"
echo "  examples      : $LIMIT"
echo "  lookup_limit  : $LOOKUP_LIMIT"
echo "  rrf_k         : $RRF_K"
if [ -n "$NAMES_FILE" ]; then
    echo "  names_file    : $NAMES_FILE"
    echo "  names_variant : $NAMES_VARIANT"
    echo "  modes         : gt_title + predicted"
else
    echo "  names_file    : (none -- gt_title mode only)"
    echo "  modes         : gt_title"
fi
echo "  output        : $OUTPUT"
echo "================================================================"

# Build optional --names-file / --names-variant args
NAMES_ARGS=()
if [ -n "$NAMES_FILE" ]; then
    NAMES_ARGS=(--names-file "$NAMES_FILE" --names-variant "$NAMES_VARIANT")
fi

uv run python "$CODE_DIR"/src/retrieval/experiments/compare_lookup_merge.py \
    --limit        "$LIMIT"        \
    --lookup-limit "$LOOKUP_LIMIT" \
    --rrf-k        "$RRF_K"        \
    --output       "$OUTPUT"       \
    "${NAMES_ARGS[@]}"

echo "================================================================"
echo "Done. Results in $OUTPUT"
echo "================================================================"
