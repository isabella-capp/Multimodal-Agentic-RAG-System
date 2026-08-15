#!/bin/bash

set -euo pipefail

[ $# -ge 1 ] || { echo "usage: scripts/submit.sh <script.sh> [sbatch args...]"; exit 1; }
SCRIPT="$1"; shift

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
[ -f "$SCRIPT" ] || { echo "no such script: $SCRIPT"; exit 1; }

RUN_ID="$(date +%Y%m%d-%H%M%S)${VARIANT:+-$VARIANT}"
SNAP="$REPO/runs/$RUN_ID"
LOG_DIR="logs/$RUN_ID"          # everything this run writes ends up here
mkdir -p "$SNAP" "$REPO/$LOG_DIR"
cp -r src scripts "$SNAP"/

mkdir -p "$SNAP/evqa_eval"
cp evqa_eval/*.py evqa_eval/pyproject.toml evqa_eval/uv.lock "$SNAP/evqa_eval"/
# The venv itself is reused from the repo — snapshotting 1.8 GB per run makes no
# sense. The lock files record what it was supposed to contain, which is enough
# to rebuild it or to notice it changed.
cp pyproject.toml uv.lock "$SNAP"/ 2>/dev/null || true

# What the snapshot corresponds to, for the times it matters later.
{
    echo "run_id:  $RUN_ID"
    echo "variant: ${VARIANT:-<none>}"
    echo "commit:  $(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "branch:  $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "dirty:   $(git status --porcelain 2>/dev/null | wc -l) files"
    echo "script:  $SCRIPT"
    echo "vllm:    $("/homes/$USER/vllm_venv/bin/vllm" --version 2>/dev/null | head -1 || echo "not built yet")"
} > "$SNAP/RUN_INFO"
git diff HEAD > "$SNAP/uncommitted.diff" 2>/dev/null || true

JOB=$(sbatch --parsable \
      --export=ALL,CODE_DIR="$SNAP",RUN_ID="$RUN_ID",LOG_DIR="$LOG_DIR" \
      --output="$LOG_DIR/%x_%j.out" --error="$LOG_DIR/%x_%j.err" \
      "$@" "$SNAP/$SCRIPT")
echo "submitted $JOB"
echo "  code: runs/$RUN_ID"
echo "  logs: $LOG_DIR/"
echo "  variant: ${VARIANT:-<none>}"
echo "$JOB" > "$SNAP/JOB_ID"
