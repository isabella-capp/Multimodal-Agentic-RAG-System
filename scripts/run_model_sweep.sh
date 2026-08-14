#!/bin/bash
#SBATCH --job-name=model_sweep
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --account=cvcs2026

# Same agent, same tools, same prompt, same examples — three model sizes.
# Isolates the model axis from the scaffold axis.
#
# Per model, two measurements:
#   naming probe — can it name the entity well enough to resolve the article?
#                  (the recall ceiling of `lookup_article`)
#   agentic eval — end-to-end BEM accuracy + tool-call behaviour
#
# The remote model needs a key in the submitting shell:
#   export LLM_API_KEY=sk-or-v1-...
#   sbatch --export=ALL scripts/run_model_sweep.sh

set -euo pipefail

LOCAL_SIZES=(3b 7b)
REMOTE_MODEL="qwen/qwen2.5-vl-72b-instruct"
REMOTE_TAG="qwen72b"
LIMIT=1000

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="/homes/$USER/vllm_venv"
TOOL_TEMPLATE="$PROJECT_DIR/scripts/qwen2.5-vl-tool-chat-template.jinja"
OUT_DIR="outputs/agentic/sweep"

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

VLLM_PID=""
STAGED_DIR=""
cleanup() {
    [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true
    [ -n "$STAGED_DIR" ] && rm -rf "$STAGED_DIR" || true
}
trap cleanup EXIT

# Evaluate one model reachable at $2 (naming probe, then end-to-end agent).
evaluate() {
    local model="$1" base_url="$2" tag="$3"
    echo "################ $tag  ($model)"
    if [ -f "$OUT_DIR/results_${tag}.json" ]; then
        echo "already scored — skipping"
        return 0
    fi

    uv run python src/agent/run_naming_probe.py \
        --model-name "$model" --base-url "$base_url" \
        --output "$OUT_DIR/naming_${tag}.jsonl" \
        --limit "$LIMIT" --concurrency 8 || true

    local pred="$OUT_DIR/predictions_${tag}.jsonl"
    uv run python src/agent/run_agentic_eval.py \
        --model-name "$model" --base-url "$base_url" \
        --output "$pred" --limit "$LIMIT" \
        --concurrency 4 --debug-samples 3 || true

    (cd "$PROJECT_DIR/evqa_eval" && uv run python score_evqa.py \
        --predictions "../$pred" --output "../$OUT_DIR/results_${tag}.json") || true
}

serve_local() {
    local size="$1" model gpu_util maxlen=()
    case "$size" in
        3b) model="Qwen/Qwen2.5-VL-3B-Instruct"; gpu_util=0.35 ;;
        7b) model="Qwen/Qwen2.5-VL-7B-Instruct"; gpu_util=0.45; maxlen=(--max-model-len 16384) ;;
    esac

    local serve="$model" ticks=120
    local snap
    snap=$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen2.5-VL-*"${size^^}"-Instruct/snapshots/*/ 2>/dev/null | head -1)
    local avail
    avail=$(df -BG --output=avail "${TMPDIR:-/tmp}" 2>/dev/null | tail -1 | tr -dc '0-9')
    if [ -n "$snap" ] && [ "${avail:-0}" -ge 20 ]; then
        STAGED_DIR="${TMPDIR:-/tmp}/qwen_vl_${size}_${SLURM_JOB_ID}"
        mkdir -p "$STAGED_DIR"
        echo "Staging $size weights to node-local disk ..."
        if cp -rL "$snap". "$STAGED_DIR"/; then serve="$STAGED_DIR"; else
            rm -rf "$STAGED_DIR"; STAGED_DIR=""; ticks=1200
        fi
    else
        ticks=1200
    fi

    "$VENV/bin/vllm" serve "$serve" --port 8000 \
        --served-model-name "$model" \
        --gpu-memory-utilization "$gpu_util" "${maxlen[@]}" \
        --enable-auto-tool-choice --tool-call-parser hermes \
        --chat-template "$TOOL_TEMPLATE" \
        --safetensors-load-strategy=prefetch \
        > "logs/sweep_srv_${size}_${SLURM_JOB_ID}.log" 2>&1 &
    VLLM_PID=$!

    echo "waiting for vLLM $size (up to $((ticks / 6)) min) ..."
    for _ in $(seq 1 "$ticks"); do
        curl -sf http://localhost:8000/health >/dev/null 2>&1 && break
        sleep 10
    done
    curl -sf http://localhost:8000/health >/dev/null 2>&1
}

stop_local() {
    [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true
    [ -n "$STAGED_DIR" ] && rm -rf "$STAGED_DIR" || true
    VLLM_PID=""; STAGED_DIR=""
    sleep 20  # let the GPU drain before the next server
}

if [ ! -x "$VENV/bin/vllm" ]; then
    uv venv "$VENV" --python 3.12 --clear
    uv pip install --python "$VENV/bin/python" "vllm==0.25.1"
fi

for size in "${LOCAL_SIZES[@]}"; do
    # Don't pay a vLLM startup for a model that is already scored.
    if [ -f "$OUT_DIR/results_qwen${size}.json" ]; then
        echo "################ qwen${size} — already scored, skipping"
        continue
    fi
    if serve_local "$size"; then
        evaluate "Qwen/Qwen2.5-VL-${size^^}-Instruct" "http://localhost:8000/v1" "qwen${size}"
    else
        echo "vLLM failed to start for $size — skipping."
    fi
    stop_local
done

if [ -n "${LLM_API_KEY:-}" ]; then
    evaluate "$REMOTE_MODEL" "https://openrouter.ai/api/v1" "$REMOTE_TAG"
else
    echo "LLM_API_KEY not set — skipping the remote model."
fi

echo "################ summary"
for f in "$OUT_DIR"/results_*.json; do
    [ -e "$f" ] && echo "--- $f" && cat "$f"
done
