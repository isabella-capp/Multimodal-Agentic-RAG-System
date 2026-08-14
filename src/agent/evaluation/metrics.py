from __future__ import annotations

import json
import statistics
import time
from collections import Counter, defaultdict

from agent.schemas import AgentRun


class MetricsCollector:
    """Accumulates the metrics of each ``AgentRun`` and summarises them.

    Beyond cost/latency, it records *how* the agent retrieves — which tool it
    enters with, how often each tool comes back empty, and which call sequences
    it settles into — so runs can be compared on behaviour, not just accuracy.
    """

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
                "paragraphs_used": run.paragraphs_used,
                "sequence": run.tool_sequence,
                "steps": [
                    {"tool": s.tool, "miss": s.miss, "num_articles": s.num_articles}
                    for s in run.steps
                ],
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
            "avg_paragraphs_read": round(
                statistics.mean(r["paragraphs_used"] for r in self._rows), 2
            ),
            "tool_usage": self._tool_usage(n),
            "first_tool_pct": self._first_tool(n),
            "top_sequences": self._top_sequences(),
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

    def _tool_usage(self, n: int) -> dict:
        calls: Counter = Counter()
        misses: Counter = Counter()
        users: dict[str, set] = defaultdict(set)
        articles: dict[str, list] = defaultdict(list)
        for i, row in enumerate(self._rows):
            for step in row["steps"]:
                tool = step["tool"]
                calls[tool] += 1
                misses[tool] += step["miss"]
                users[tool].add(i)
                articles[tool].append(step["num_articles"])
        return {
            tool: {
                "calls": c,
                "examples_using_pct": round(100 * len(users[tool]) / n, 1),
                "miss_pct": round(100 * misses[tool] / c, 1),
                "avg_articles_returned": round(statistics.mean(articles[tool]), 2),
            }
            for tool, c in calls.most_common()
        }

    def _first_tool(self, n: int) -> dict:
        first = Counter(r["sequence"][0] if r["sequence"] else "<none>" for r in self._rows)
        return {tool: round(100 * c / n, 1) for tool, c in first.most_common()}

    def _top_sequences(self, limit: int = 8) -> dict:
        seqs = Counter(" → ".join(r["sequence"]) or "<none>" for r in self._rows)
        return dict(seqs.most_common(limit))
