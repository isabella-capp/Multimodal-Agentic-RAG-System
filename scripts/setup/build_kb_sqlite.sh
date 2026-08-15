#!/bin/bash
#SBATCH --job-name=build_kb_sqlite
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/homes/%u/cvcs2026/logs/build_kb_%j.out
#SBATCH --error=/homes/%u/cvcs2026/logs/build_kb_%j.err
#SBATCH --account=cvcs2026

set -euo pipefail

PROJECT_DIR="/homes/$USER/cvcs2026"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
unset SSL_CERT_DIR

cd "$PROJECT_DIR"
mkdir -p logs

uv run python src/retrieval/build_kb_sqlite.py "$@"
