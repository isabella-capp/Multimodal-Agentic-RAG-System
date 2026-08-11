#!/bin/bash
#SBATCH --job-name=agentic_vllm
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/agentic_vllm_%j.out
#SBATCH --error=logs/agentic_vllm_%j.err
#SBATCH --account=cvcs2026

# Serve Qwen2.5-VL on vLLM and run the agentic `research` pipeline (forced
# retrieve → multimodal extractor sub-agent → answer, with a cross-encoder refine
# tool). vLLM + EVA-CLIP + KB + reranker load once; predictions are scored at the end.
#
# Submit from the repository root:  sbatch scripts/run_agentic_vllm.sh [3b|7b]
# (default: 3b). Log paths are relative to the submit dir, so `logs/` must exist
# (it is tracked via logs/.gitkeep).

set -euo pipefail

SIZE="${1:-3b}"
case "$SIZE" in
    3b) MODEL="Qwen/Qwen2.5-VL-3B-Instruct"; GPU_UTIL=0.40; MAXLEN=() ;;
    7b) MODEL="Qwen/Qwen2.5-VL-7B-Instruct"; GPU_UTIL=0.50; MAXLEN=(--max-model-len 16384) ;;
    *)  echo "Unknown size '$SIZE' (use 3b or 7b)"; exit 1 ;;
esac

# Repo root = where sbatch was invoked (fallback: this script's parent dir).
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="/homes/$USER/vllm_venv"  # per-user, kept off /work (too slow to load from)
PRED="outputs/agentic/predictions_agentic_${SIZE}_research.jsonl"
RESULT="outputs/agentic/results_agentic_${SIZE}_research.json"
TOOL_TEMPLATE="$PROJECT_DIR/scripts/qwen2.5-vl-tool-chat-template.jinja"

export HF_HOME="/work/cvcs2026/recursive_retrievers/hf_cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$HOME/.local/bin:$PATH"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/cvcs2026/$USER/.uv_cache}"
mkdir -p "$UV_CACHE_DIR"
unset SSL_CERT_DIR
cd "$PROJECT_DIR"
mkdir -p logs outputs/agentic

# Build the isolated vLLM venv on first use (kept on /homes; /work is too slow).
if [ ! -x "$VENV/bin/vllm" ]; then
    echo "Creating vLLM venv at $VENV ..."
    uv venv "$VENV" --python 3.12 --clear
    uv pip install --python "$VENV/bin/python" "vllm==0.25.1"
fi

SERVE_MODEL="$MODEL"        # what vLLM loads (local path if staging succeeds)
STAGED_DIR=""               # cleaned up on exit
WAIT_TICKS=120              # health-check window (×10s); widened on fallback
SRC_SNAP=$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen2.5-VL-*"${SIZE^^}"-Instruct/snapshots/*/ 2>/dev/null | head -1)
LOCAL_ROOT="${TMPDIR:-/tmp}"
NEED_GB=20
AVAIL_GB=$(df -BG --output=avail "$LOCAL_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')

if [ -n "$SRC_SNAP" ] && [ "${AVAIL_GB:-0}" -ge "$NEED_GB" ]; then
    STAGED_DIR="$LOCAL_ROOT/qwen_vl_${SIZE}_${SLURM_JOB_ID}"
    echo "Staging weights: $SRC_SNAP -> $STAGED_DIR (local ${AVAIL_GB}G free) ..."
    mkdir -p "$STAGED_DIR"
    if cp -rL "$SRC_SNAP". "$STAGED_DIR"/ ; then
        SERVE_MODEL="$STAGED_DIR"
        echo "Staging done; serving from local disk."
    else
        echo "Staging copy failed; falling back to /work (wider wait window)."
        rm -rf "$STAGED_DIR"; STAGED_DIR=""; WAIT_TICKS=1200
    fi
else
    echo "Cannot stage (snapshot='$SRC_SNAP', local free=${AVAIL_GB:-?}G < ${NEED_GB}G); serving from /work with wider wait window."
    WAIT_TICKS=1200
fi

"$VENV/bin/vllm" serve "$SERVE_MODEL" --port 8000 \
    --served-model-name "$MODEL" \
    --gpu-memory-utilization "$GPU_UTIL" "${MAXLEN[@]}" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --chat-template "$TOOL_TEMPLATE" \
    --safetensors-load-strategy=prefetch \
    > "logs/agentic_vllm_srv_${SIZE}_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true; [ -n "$STAGED_DIR" ] && rm -rf "$STAGED_DIR" || true' EXIT

echo "waiting for vLLM (up to $((WAIT_TICKS * 10 / 60)) min) ..."
for _ in $(seq 1 "$WAIT_TICKS"); do
    curl -sf http://localhost:8000/health >/dev/null 2>&1 && break
    sleep 10
done
curl -sf http://localhost:8000/health >/dev/null 2>&1 || { echo "vLLM failed to start"; exit 1; }
echo "vLLM ready"

uv run python src/agent/run_agentic_eval.py \
    --model-name "$MODEL" --output "$PRED" --pipeline research \
    --concurrency 4 --debug-samples 5

export TFHUB_CACHE_DIR="/work/cvcs2026/recursive_retrievers/tfhub_cache"
cd "$PROJECT_DIR/evqa_eval"
uv run python score_evqa.py --predictions "../$PRED" --output "../$RESULT" || true
