#!/bin/bash
#SBATCH --job-name=baseline_b
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/baseline_b_%j.out
#SBATCH --error=logs/baseline_b_%j.err
#SBATCH --account=cvcs2026
#
# Setting B alone, for iterating on the retrieval baseline without re-running A
# and C. scripts/run_abc.sh stays the reference table.
#
#   VARIANT=k20n30 scripts/submit.sh scripts/baselines/run_b.sh
#
# LEGACY=1 drops the shared answer-format block, reproducing the prompt the
# historical B=0.401 used. That run averaged 19.3 words per answer against 2.4
# now, while containing the correct answer just as often (0.308 vs 0.297): the
# gap looks like a regression but is BEM paying for length. This flag is how we
# show that rather than assert it.
#
#   LEGACY=1 VARIANT=legacy-prompt scripts/submit.sh scripts/baselines/run_b.sh

set -euo pipefail

MODEL="Qwen/Qwen3-VL-8B-Instruct"
TAG="qwen3vl8b"
GPU_UTIL=0.50
MAX_LEN=32768
TOP_K=20
TOP_N=20
CONCURRENCY=8

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV="/homes/$USER/vllm_venv"
CODE_DIR="${CODE_DIR:-$PROJECT_DIR}"
OUT_DIR="outputs/baselines/$TAG/${RUN_ID:-manual}"

PROMPT=()
[ "${LEGACY:-0}" = "1" ] && PROMPT=(--legacy-prompt)
if [ "${SMOKE:-0}" = "1" ]; then
    LIMIT=(--limit 5); DEBUG=5; OUT_DIR="$OUT_DIR/smoke"
else
    LIMIT=(); DEBUG=10
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
mkdir -p "${LOG_DIR:-logs}" "$OUT_DIR"
source "$CODE_DIR/scripts/lib/vllm.sh"

ensure_vllm_venv
serve_model "$MODEL" "$GPU_UTIL" "$MAX_LEN"

echo "################ B — retrieval (top-k=$TOP_K top-n=$TOP_N)${LEGACY:+, legacy prompt}"
uv run python "$CODE_DIR"/src/vlm/run_inference.py \
    --model-name "$MODEL" --base-url "$BASE_URL" \
    --output "$OUT_DIR/predictions_B.jsonl" \
    --use-retrieval --top-k "$TOP_K" --rerank-top-n "$TOP_N" \
    --concurrency "$CONCURRENCY" --debug-samples "$DEBUG" \
    "${PROMPT[@]}" "${LIMIT[@]}"

stop_model

echo "################ scoring"
(cd "$PROJECT_DIR/evqa_eval" && uv run python "$CODE_DIR/evqa_eval/score_evqa.py" \
    --predictions "../$OUT_DIR/predictions_B.jsonl" \
    --output "../$OUT_DIR/results_B.json") || true

echo "################ summary"
python3 -c "
import json, statistics
r = json.load(open('$OUT_DIR/results_B.json'))
rows = [json.loads(l) for l in open('$OUT_DIR/predictions_B.jsonl')]
words = statistics.mean(len((x.get('prediction') or '').split()) for x in rows)
print(f\"  B: {r['accuracy_overall']}   ({words:.1f} words per answer)\")
print('  by type:', r['accuracy_by_type'])
" 2>/dev/null || echo "  n/a"
cat "$CODE_DIR/RUN_INFO" 2>/dev/null || echo "code: live tree"
