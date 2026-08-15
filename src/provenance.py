from __future__ import annotations

import json
import os
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _git(*args) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def stamp(output: str, **extra) -> dict:
    """Record which code produced a predictions file, next to it as .meta.json.

    Outputs are git-ignored, so without this there is no way back from a result
    to the code that made it — which matters as soon as two people try different
    strategies in parallel. ``dirty`` means the run used uncommitted changes and
    is therefore not reproducible from the commit alone.
    """
    meta = {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "command": " ".join(sys.argv),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **extra,
    }
    with open(output.rsplit(".", 1)[0] + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if meta["dirty"]:
        print("WARNING: uncommitted changes — this run is not reproducible from "
              f"commit {meta['commit']}")
    return meta
