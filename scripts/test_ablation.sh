#!/bin/bash
#SBATCH --job-name=test_ablation
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/homes/%u/cvcs2026/logs/test_ablation_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/test_ablation_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail

PROJECT_DIR="/homes/$USER/cvcs2026"

export HF_HOME="/work/cvcs2026/recursive_retrievers/hf_cache/huggingface"
export HF_HUB_OFFLINE=1
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
unset SSL_CERT_DIR

cd "$PROJECT_DIR"
mkdir -p logs outputs/test_ablation

VAL_JSON="/work/cvcs2026/encyclopedic/encyclopedic_val_split.json"
if [ ! -f "$VAL_JSON" ]; then
    echo "Creating validation split …"
    uv run python src/ablation/create_val_split.py
fi

echo "Running FAST ABLATION TEST (2 samples, 2 configs) …"
uv run python src/ablation/run_ablation_cross.py \
    --val-json "$VAL_JSON" \
    --output-dir outputs/test_ablation \
    --top-k-values 5 10 \
    --rerank-top-n-values 5 \
    --limit 2 \
    --debug-samples 2

echo "Test complete! Check outputs/test_ablation for results."
