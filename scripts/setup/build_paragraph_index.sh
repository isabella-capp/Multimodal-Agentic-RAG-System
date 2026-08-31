#!/bin/bash
#SBATCH --job-name=para_fts
#SBATCH --partition=all_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/para_fts_%j.out
#SBATCH --error=logs/para_fts_%j.err
#SBATCH --account=cvcs2026
#
# Full-text index over the paragraph text of all 2.0M articles (14.1M rows).
#
#   scripts/submit.sh scripts/setup/build_paragraph_index.sh
#
# No GPU: this is SQLite only, so it does not compete for the boost queue.
#
# It writes into the KB itself, so one file stays the source of truth. Two
# consequences worth knowing before launching it:
#
#   - it takes a write lock on the 18.6 GB file the whole group reads, so submit
#     it when nothing else is running, or with
#     --dependency=afterany:<jobs> on whatever is reading;
#   - readers must not open the KB with immutable=1 any more, since that flag
#     promises SQLite the bytes never change. KnowledgeBase drops it.
#
# Why it exists: our two ways into the KB both go through the model — the image
# index needs the entity to have a photograph (40.6% recall@20) and the name
# lookup needs the model to name it (11.6% with Qwen3-VL-8B). This one is
# queried with the question itself.

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CODE_DIR="${CODE_DIR:-$PROJECT_DIR}"

export PYTHONUNBUFFERED=1
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/work/cvcs2026/recursive_retrievers/$USER/.uv_cache}"
mkdir -p "$UV_CACHE_DIR"
unset SSL_CERT_DIR

cd "$PROJECT_DIR"
mkdir -p "${LOG_DIR:-logs}"

uv run python "$CODE_DIR"/src/retrieval/build_kb_sqlite.py --paragraphs-fts "$@"
