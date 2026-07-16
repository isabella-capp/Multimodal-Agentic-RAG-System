from __future__ import annotations

import json
import statistics
import time
from collections import Counter

from agent.schemas import AgentRun


class MetricsCollector:
    """Accumulates the metrics of each ``AgentRun`` and summarises them."""

    def __init__(self):
        self._rows: list[dict] = []
        self._t0 = time.time()

    def record(self, run: AgentRun) -> None:
        self._rows.append(
            {
                "tool_called": run.tool_called,
                "num_tool_calls": run.num_tool_calls,
                "elapsed_seconds": run.elapsed_seconds,
                "error": run.error,
            }
        )

    def summary(self) -> dict:
        n = len(self._rows)
        wall = time.time() - self._t0
        if n == 0:
            return {"examples": 0, "wall_seconds": round(wall, 1)}
        calls = [r["num_tool_calls"] for r in self._rows]
        elapsed = [r["elapsed_seconds"] for r in self._rows]
        called = sum(1 for r in self._rows if r["tool_called"])
        errors = [r["error"] for r in self._rows if r["error"]]
        return {
            "examples": n,
            "tool_called": called,
            "tool_called_pct": round(100 * called / n, 1),
            "num_tool_calls_distribution": dict(sorted(Counter(calls).items())),
            "avg_tool_calls": round(statistics.mean(calls), 2),
            "avg_seconds_per_example": round(statistics.mean(elapsed), 2),
            "median_seconds_per_example": round(statistics.median(elapsed), 2),
            "errors": len(errors),
            "error_types": dict(Counter(errors)),
            "wall_seconds": round(wall, 1),
            "throughput_per_min": round(60 * n / wall, 1) if wall else None,
        }

    def save(self, path: str) -> dict:
        summary = self.summary()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary
