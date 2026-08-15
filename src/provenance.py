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


def _snapshot() -> dict | None:
    """Read RUN_INFO if this is running from a scripts/submit.sh snapshot."""
    path = os.path.join(REPO, "RUN_INFO")
    if not os.path.exists(path):
        return None
    info = {}
    with open(path) as f:
        for line in f:
            key, _, value = line.partition(":")
            info[key.strip()] = value.strip()
    return info


def stamp(output: str, **extra) -> dict:
    """Record which code produced a predictions file, next to it as .meta.json.

    Outputs are git-ignored, so without this there is no way back from a result
    to the code that made it.

    Under scripts/submit.sh the answer comes from the snapshot's RUN_INFO, which
    was written at SUBMIT time. Reading git here instead would report whatever
    the working tree holds when the job finally starts — which is exactly what
    changes while a job sits in the queue, so it would quietly record the wrong
    commit for the run it is describing.
    """
    snap = _snapshot()
    meta = {
        "run_id": (snap or {}).get("run_id"),
        "variant": (snap or {}).get("variant"),
        "code": REPO if snap else None,
        "commit": snap["commit"] if snap else _git("rev-parse", "HEAD"),
        "branch": snap["branch"] if snap else _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": (snap["dirty"] != "0 files") if snap else bool(_git("status", "--porcelain")),
        "command": " ".join(sys.argv),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **extra,
    }
    with open(output.rsplit(".", 1)[0] + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if meta["dirty"] and not snap:
        print("WARNING: uncommitted changes and no code snapshot — this run is "
              f"not reproducible from commit {meta['commit']}. "
              "Submit with scripts/submit.sh to freeze the code.")
    return meta
