#!/bin/bash
#SBATCH --job-name=naming_crops
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/naming_crops_%j.out
#SBATCH --error=logs/naming_crops_%j.err
#SBATCH --account=cvcs2026
#
# Does cropping help the model name what it sees?
#
#   scripts/submit.sh scripts/agentic/run_naming_crops.sh
#   VARIANTS=full,center80,center60,center40 scripts/submit.sh …
#   BOXES=outputs/retrieval/boxes.jsonl VARIANTS=full,box scripts/submit.sh …
#
# Naming is the gate on the only channel with headroom — 84.2% of gold titles
# are reachable by name against 40.6% recall@20 by image — and Qwen3-VL-8B
# reaches it 11.6% of the time.
#
# It is not looking at the background: it answers `Gila monster` for a `Tiliqua
# rugosa` and `Burg Hohenölsen` for an `Uzhhorod Castle`, so the subject is
# found and the instance is wrong. A crop can then only pay by magnifying, and
# the probe rescales every crop back to the original size so the token budget
# stays fixed and the magnification is what changes. UPSCALE=0 runs it without
# that, which should be flat or worse if the hypothesis holds.
#
# No retrieval and no answer generation: 32 output tokens per image.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
TAG="${TAG:-qwen3vl8b}"
GPU_UTIL=0.85          # nothing else on this GPU: no retriever, no reranker
MAX_LEN=32768
VARIANTS="${VARIANTS:-full,center80,center60,center40}"
UPSCALE="${UPSCALE:-1}"
LIMIT="${LIMIT:-1000}"
CONCURRENCY=8

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV="/homes/$USER/vllm_venv"
CODE_DIR="${CODE_DIR:-$PROJECT_DIR}"
OUT_DIR="outputs/agentic/$TAG/${RUN_ID:-manual}"

FLAGS=()
[ "$UPSCALE" = "0" ] && FLAGS+=(--no-upscale)
[ -n "${BOXES:-}" ] && FLAGS+=(--boxes "$BOXES")

export HF_HOME="/work/cvcs2026/recursive_retrievers/hf_cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/cvcs2026/recursive_retrievers/$USER/.uv_cache}"
mkdir -p "$UV_CACHE_DIR"
unset SSL_CERT_DIR

cd "$PROJECT_DIR"
mkdir -p "${LOG_DIR:-logs}" "$OUT_DIR"
source "$CODE_DIR/scripts/lib/vllm.sh"

echo "variants: $VARIANTS   upscale: $UPSCALE   boxes: ${BOXES:-<none>}"
ensure_vllm_venv
serve_model "$MODEL" "$GPU_UTIL" "$MAX_LEN"

uv run python "$CODE_DIR"/src/agent/experiments/naming_probe.py \
    --model-name "$MODEL" --base-url "$BASE_URL" \
    --output "$OUT_DIR/naming_crops.jsonl" \
    --variants "$VARIANTS" --limit "$LIMIT" --concurrency "$CONCURRENCY" \
    "${FLAGS[@]}"

stop_model
cat "$CODE_DIR/RUN_INFO" 2>/dev/null || echo "code: live tree"
