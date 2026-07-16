#!/bin/bash
#SBATCH --job-name=agentic_vllm
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/homes/%u/cvcs2026/logs/agentic_vllm_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/agentic_vllm_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail

PROJECT_DIR="/homes/$USER/cvcs2026"
VENV="/homes/$USER/vllm_venv"
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
PRED="outputs/predictions_agentic_auto.jsonl"

export HF_HOME="/work/cvcs2026/recursive_retrievers/hf_cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$HOME/.local/bin:$PATH"
unset SSL_CERT_DIR
cd "$PROJECT_DIR"
mkdir -p logs outputs

# Build the isolated vLLM venv on first use (kept on /homes; /work is too slow).
if [ ! -x "$VENV/bin/vllm" ]; then
    echo "Creating vLLM venv at $VENV ..."
    uv venv "$VENV" --python 3.12
    uv pip install --python "$VENV/bin/python" "vllm==0.25.1"
fi

# vLLM server in background; capped GPU memory to share the card with retrieval.
"$VENV/bin/vllm" serve "$MODEL" --port 8000 \
    --gpu-memory-utilization 0.40 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --safetensors-load-strategy=prefetch \
    > "logs/agentic_vllm_srv_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

echo "waiting for vLLM ..."
for _ in $(seq 1 120); do
    curl -sf http://localhost:8000/health >/dev/null 2>&1 && break
    sleep 10
done
curl -sf http://localhost:8000/health >/dev/null 2>&1 || { echo "vLLM failed to start"; exit 1; }
echo "vLLM ready"

uv run python src/agent/run_agentic_eval.py \
    --output "$PRED" --concurrency 4 --limit 200 --debug-samples 5

export TFHUB_CACHE_DIR="/work/cvcs2026/recursive_retrievers/tfhub_cache"
cd "$PROJECT_DIR/evqa_eval"
uv run python score_evqa.py --predictions "../$PRED" --output "../outputs/results_agentic_auto.json" || true
