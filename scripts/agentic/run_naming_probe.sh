#!/bin/bash
#SBATCH --job-name=naming_probe
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/naming_%j.out
#SBATCH --error=logs/naming_%j.err
#SBATCH --account=cvcs2026

# How often can the model name the entity in the image well enough to resolve the
# right Wikipedia article? That rate is the recall ceiling of `lookup_article`.
#
# The model is running locally via a vLLM server.
# Submit directly with: sbatch nome_dello_script.sh

set -euo pipefail

MODEL="Qwen/Qwen3-VL-8B-Instruct"
TAG="qwen3vl8b"
LIMIT=1000

# vLLM params
GPU_UTIL=0.90      # Il modello occupa l'intera GPU per l'inferenza
MAX_LEN=32768

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV="/homes/$USER/vllm_venv"
CODE_DIR="${CODE_DIR:-$PROJECT_DIR}"

export HF_HOME="/work/cvcs2026/recursive_retrievers/hf_cache/huggingface"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/cvcs2026/recursive_retrievers/$USER/.uv_cache}"
mkdir -p "$UV_CACHE_DIR"
unset SSL_CERT_DIR

# Molti client richiedono che la chiave sia presente, anche se fittizia,
# per non dare errore in fase di inizializzazione locale.
export LLM_API_KEY="dummy-local-key"
export OPENAI_API_KEY="dummy-local-key"

cd "$PROJECT_DIR"
mkdir -p "${LOG_DIR:-logs}" outputs/agentic

source "$CODE_DIR/scripts/lib/vllm.sh"

echo "Avvio dell'ambiente e del server vLLM..."
ensure_vllm_venv
serve_model "$MODEL" "$GPU_UTIL" "$MAX_LEN"

echo "Esecuzione di naming_probe.py..."
uv run python "$CODE_DIR"/src/agent/experiments/naming_probe.py \
    --model-name "$MODEL" --base-url "$BASE_URL" \
    --output "outputs/agentic/naming_probe_${TAG}.jsonl" \
    --variants full \
    --limit "$LIMIT" \
    --concurrency 8

stop_model
echo "Finito."