#!/bin/bash
#SBATCH --job-name=prime_df
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=logs/prime_df_%j.out
#SBATCH --error=logs/prime_df_%j.err
#SBATCH --account=cvcs2026
#
# Fill the term-frequency cache the full-text channel needs, once, on CPU.
#
#   scripts/submit.sh scripts/retrieval/run_prime_df.sh
#
# Without it every GPU job re-measures each word from cold: the recall probe
# spent an hour on the cluster against six minutes locally, all of it in first
# touches of ~4k terms. The counts only change when the KB is rebuilt.

set -euo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CODE_DIR="${CODE_DIR:-$PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/cvcs2026/recursive_retrievers/$USER/.uv_cache}"
mkdir -p "$UV_CACHE_DIR"
unset SSL_CERT_DIR
cd "$PROJECT_DIR"; mkdir -p "${LOG_DIR:-logs}" outputs/retrieval
uv run python "$CODE_DIR"/src/retrieval/experiments/prime_df_cache.py "$@"
