#!/bin/bash
#SBATCH --job-name=ablation_cross
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/homes/%u/cvcs2026/logs/ablation_cross_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/ablation_cross_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail

PROJECT_DIR="/homes/$USER/cvcs2026"

export HF_HOME="/work/cvcs2026/recursive_retrievers/hf_cache/huggingface"
export HF_HUB_OFFLINE=1
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
unset SSL_CERT_DIR

cd "$PROJECT_DIR"
mkdir -p logs outputs/ablation data

# ── Step 1: Create validation split (if not already done) ────────────────
VAL_JSON="$PROJECT_DIR/data/encyclopedic_val_split.json"
if [ ! -f "$VAL_JSON" ]; then
    echo "Creating validation split …"
    uv run python src/ablation/create_val_split.py
    echo ""
fi

# ── Step 2: Run ablation study ───────────────────────────────────────────
uv run python src/ablation/run_ablation_cross.py \
    --val-json "$VAL_JSON" \
    --output-dir outputs/ablation \
    --top-k-values 5 10 20 50 \
    --rerank-top-n-values 5 10 20 30 \
    --debug-samples 1

# ── Step 3: Run BEM evaluation on each config ───────────────────────────
echo ""
echo "Running BEM evaluation on all ablation predictions …"

cd "$PROJECT_DIR/evqa_eval"
export TFHUB_CACHE_DIR="/work/cvcs2026/recursive_retrievers/tfhub_cache"

for pred_file in "$PROJECT_DIR"/outputs/ablation/predictions_cross_*.jsonl; do
    basename=$(basename "$pred_file" .jsonl)
    result_file="$PROJECT_DIR/outputs/ablation/results_BEM_${basename#predictions_}.json"

    if [ -f "$result_file" ]; then
        echo "  [skip] $result_file already exists"
        continue
    fi

    echo "  Scoring $basename …"
    uv run python score_evqa.py \
        --predictions "$pred_file" \
        --output "$result_file"
done

# ── Step 4: Aggregate final results ─────────────────────────────────────
cd "$PROJECT_DIR"
echo ""
echo "Aggregating results …"
uv run python src/ablation/aggregate_ablation.py \
    --results-dir outputs/ablation \
    --output outputs/ablation/ablation_summary_BEM.json
