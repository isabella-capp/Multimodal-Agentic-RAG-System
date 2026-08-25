#!/bin/bash
#SBATCH --job-name=rerankers
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/rerankers_%j.out
#SBATCH --error=logs/rerankers_%j.err
#SBATCH --account=cvcs2026
#
# Is the cross-encoder the bottleneck? Measured without any generation: on the
# examples where the right article IS pooled and some paragraph DOES state the
# answer, how often does the reranker put that paragraph in the top-n?
#
# If the answer is ~90% the reranker is fine and a bigger one buys nothing; B's
# 0.528 accuracy on pool-hit would then be the reader's limit, not retrieval's.
#
#   scripts/submit.sh scripts/retrieval/run_compare_rerankers.sh

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CODE_DIR="${CODE_DIR:-$PROJECT_DIR}"

export HF_HOME="/work/cvcs2026/recursive_retrievers/hf_cache/huggingface"
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
unset SSL_CERT_DIR       # bge-reranker-v2-m3 may need downloading; compute nodes have internet

cd "$PROJECT_DIR"
mkdir -p "${LOG_DIR:-logs}"

uv run python "$CODE_DIR"/src/retrieval/experiments/compare_rerankers.py --limit 300 --top-k 20
