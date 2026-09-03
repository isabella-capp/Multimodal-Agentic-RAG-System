"""Ablation: category-priority merge vs rank-based (RRF) merge in lookup_articles.

Two evaluation modes
--------------------

``gt_title`` (ceiling / perfect namer)
    The ground-truth Wikipedia title is used as the name query.  This
    measures the *ceiling* of each merge strategy -- what it could achieve
    with a perfect entity namer.  Always computed (backward-compatible).

``predicted`` (realistic / production)
    VLM-predicted names from a ``naming_probe`` JSONL are used instead.
    At ~11.6% naming accuracy, names are noisy and the relative advantage
    of the merge strategies may differ significantly from the ceiling.
    Enabled by passing ``--names-file``.

    Examples where the VLM produced no name (``predicted_name`` is null)
    are excluded from per-strategy counts but tracked as ``no_name`` misses,
    because in production the agent cannot call ``lookup_article`` for them.

Both modes run in a single process; only the query name differs.
Running without ``--names-file`` reproduces old behaviour exactly.

Four strategies compared in each mode:
  ``title_only``  -- ``_fuzzy`` (Jaccard on title tokens) alone.
  ``intro_only``  -- ``_fuzzy_intro`` (BM25 on first-section text) alone.
  ``category``    -- current: title hits first, then intro-only extras.
  ``rrf``         -- Reciprocal Rank Fusion of the two ranked lists.

Run on the cluster (no GPU needed -- pure SQLite):
    # Ceiling only (same as before):
    python src/retrieval/experiments/compare_lookup_merge.py \
        --limit 1000 --output outputs/retrieval/lookup_merge_ablation.json

    # Both modes (ceiling + realistic VLM names from naming_probe):
    python src/retrieval/experiments/compare_lookup_merge.py \
        --limit 1000 \
        --names-file outputs/agentic/naming_probe.jsonl \
        --names-variant full \
        --output outputs/retrieval/lookup_merge_ablation_both.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, SRC_ROOT)

from tqdm import tqdm

import paths
from retrieval.knowledge_base import KnowledgeBase, normalize
from vlm.dataset import load_dataset

KS = (1, 3, 5)
STRATEGIES = ("title_only", "intro_only", "category", "rrf")


def norm_url(url: str) -> str:
    return (url or "").rstrip("/").lower().replace("http://", "https://")


def rrf_merge(
    title_hits: list[dict],
    intro_hits: list[dict],
    limit: int,
    rrf_k: int = 60,
) -> list[dict]:
    """Fuse two ranked lists with Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    by_url: dict[str, dict] = {}
    for rank, h in enumerate(title_hits, 1):
        url = h["wiki_url"]
        scores[url] = scores.get(url, 0.0) + 1.0 / (rrf_k + rank)
        by_url.setdefault(url, h)
    for rank, h in enumerate(intro_hits, 1):
        url = h["wiki_url"]
        scores[url] = scores.get(url, 0.0) + 1.0 / (rrf_k + rank)
        by_url.setdefault(url, h)
    ranked = sorted(scores, key=lambda u: scores[u], reverse=True)
    return [by_url[u] for u in ranked[:limit]]


def category_merge(
    title_hits: list[dict],
    intro_hits: list[dict],
    limit: int,
) -> list[dict]:
    """Current behaviour: title hits first, then intro-only extras."""
    seen = {h["wiki_url"] for h in title_hits}
    merged = list(title_hits)
    for h in intro_hits:
        if h["wiki_url"] not in seen:
            merged.append(h)
            seen.add(h["wiki_url"])
    return merged[:limit]


def recall_at_k(results: list[dict], gt_url: str, k: int) -> bool:
    return any(norm_url(h["wiki_url"]) == gt_url for h in results[:k])


def load_predicted_names(path: str, variant: str) -> dict[str, str | None]:
    """Load {unique_id: predicted_name} from a naming_probe JSONL.

    Only rows matching ``variant`` are kept.  ``predicted_name`` may be
    None when the VLM did not produce a name for that example.
    """
    names: dict[str, str | None] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("variant") == variant:
                names[row["unique_id"]] = row.get("predicted_name")
    return names


def _evaluate_mode(
    dataset: list[dict],
    kb: KnowledgeBase,
    lookup_limit: int,
    rrf_k: int,
    *,
    name_override: dict[str, str | None] | None = None,
    mode_label: str = "gt_title",
    desc: str = "lookup ablation",
) -> dict:
    """Run the four-strategy ablation for one name-query mode.

    Parameters
    ----------
    name_override:
        ``{unique_id: name}`` used instead of GT Wikipedia title.
        A ``None`` value means the VLM produced no name; counted as
        ``no_name`` miss and excluded from recall denominators so figures
        remain comparable across modes.
    """
    hits: dict[str, dict[int, int]] = {s: {k: 0 for k in KS} for s in STRATEGIES}
    n_exact = n_no_name = n_total = 0
    per_example: list[dict] = []

    for item in tqdm(dataset, desc=desc):
        uid     = item["unique_id"]
        gt_url  = norm_url(item["wikipedia_url"])
        n_total += 1

        # Resolve query name
        if name_override is not None:
            name = name_override.get(uid)
            if name is None:
                n_no_name += 1
                per_example.append({
                    "unique_id": uid, "mode": mode_label,
                    "name": None, "gt_url": gt_url, "no_name": True,
                })
                continue
        else:
            name = item["wikipedia_title"]

        # Exact alias match: same for all strategies
        exact_hits = kb.lookup_articles(name, limit=1)
        if exact_hits and exact_hits[0].get("match") == "exact":
            n_exact += 1
            # Controllo vitale aggiunto qui:
            is_correct = (norm_url(exact_hits[0]["wiki_url"]) == gt_url)
            for s in STRATEGIES:
                for k in KS:
                    hits[s][k] += is_correct
            per_example.append({
                "unique_id": uid, "mode": mode_label,
                "name": name, "gt_url": gt_url, "exact": True,
            })
            continue

        # Fuzzy sources (bypass merged lookup_articles to test strategies)
        query      = normalize(name)
        title_hits = kb._fuzzy(query, lookup_limit)
        intro_hits = kb._fuzzy_intro(query, lookup_limit)

        results_map = {
            "title_only": title_hits[:lookup_limit],
            "intro_only": intro_hits[:lookup_limit],
            "category":   category_merge(title_hits, intro_hits, lookup_limit),
            "rrf":        rrf_merge(title_hits, intro_hits, lookup_limit, rrf_k=rrf_k),
        }

        row: dict = {
            "unique_id": uid, "mode": mode_label,
            "name": name, "gt_url": gt_url, "exact": False,
            "title_hits": [h["wiki_url"] for h in title_hits],
            "intro_hits":  [h["wiki_url"] for h in intro_hits],
        }
        for s, res in results_map.items():
            for k in KS:
                ok = recall_at_k(res, gt_url, k)
                hits[s][k] += ok
            row[f"{s}_recall"] = {str(k): recall_at_k(res, gt_url, k) for k in KS}
        per_example.append(row)

    n_with_name = n_total - n_no_name
    n_fuzzy     = n_with_name - n_exact

    return {
        "mode":        mode_label,
        "n_total":     n_total,
        "n_with_name": n_with_name,
        "n_no_name":   n_no_name,
        "n_exact":     n_exact,
        "n_fuzzy":     n_fuzzy,
        "lookup_limit": lookup_limit,
        "rrf_k":       rrf_k,
        "recall": {
            s: {str(k): round(hits[s][k] / n_with_name, 4) if n_with_name else 0.0
                for k in KS}
            for s in STRATEGIES
        },
        "per_example": per_example,
        "_hits_raw":   hits,
    }


def _print_mode_report(result: dict) -> None:
    mode        = result["mode"]
    n_total     = result["n_total"]
    n_with_name = result["n_with_name"]
    n_no_name   = result["n_no_name"]
    n_exact     = result["n_exact"]
    n_fuzzy     = result["n_fuzzy"]

    print("=" * 70)
    if mode == "gt_title":
        print(f"MODE: gt_title (ceiling -- perfect namer)   n={n_total}")
    else:
        no_pct = 100 * n_no_name / n_total if n_total else 0
        print(f"MODE: predicted (realistic VLM names)   n={n_total}   "
              f"no_name={n_no_name} ({no_pct:.1f}%)")
        print(f"  evaluated on {n_with_name} examples that had a predicted name")
    print(f"  exact={n_exact}  fuzzy={n_fuzzy}  |  lookup_limit={result['lookup_limit']}")
    print()
    header = f"{'strategy':14s}" + "".join(f"  recall@{k}" for k in KS)
    print(header)
    print("-" * len(header))
    for s in STRATEGIES:
        marker = "  <- current" if s == "category" else ""
        recall = result["recall"][s]
        row = (f"{s:14s}"
               + "".join(f"  {100 * recall[str(k)]:7.1f}%" for k in KS)
               + marker)
        print(row)
    print()

    rrf_beats = category_beats = tie = 0
    for row in result["per_example"]:
        if row.get("exact") or row.get("no_name"):
            continue
        r_rrf = row.get("rrf_recall", {}).get("1", False)
        r_cat = row.get("category_recall", {}).get("1", False)
        if r_rrf and not r_cat:
            rrf_beats += 1
        elif r_cat and not r_rrf:
            category_beats += 1
        else:
            tie += 1
    print(f"recall@1 breakdown -- fuzzy cases only (n={n_fuzzy}):")
    print(f"  rrf wins over category : {rrf_beats}")
    print(f"  category wins over rrf : {category_beats}")
    print(f"  tie                    : {tie}")
    print("=" * 70)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output", default="outputs/retrieval/lookup_merge_ablation.json")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--lookup-limit", type=int, default=5)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--skip-no-intro-fts", action="store_true")
    p.add_argument("--names-file", default=None,
                   help="Path to naming_probe JSONL.  When supplied, a second "
                        "pass uses VLM-predicted names (realistic production "
                        "scenario).  Absent: only gt_title mode runs "
                        "(backward-compatible).")
    p.add_argument("--names-variant", default="full",
                   help="naming_probe variant to use (default: 'full').  "
                        "Other options: 'center80', 'center60', 'box'.")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    kb = KnowledgeBase(paths.KB_PATH)

    if not kb._has_text_index:
        print("WARNING: paragraphs_fts absent -- intro_only and rrf return [].")
        if args.skip_no_intro_fts:
            print("Aborting (--skip-no-intro-fts).")
            return

    dataset = load_dataset(paths.JSON_PATH, paths.BASE_FOLDER)[: args.limit]

    # --- Mode 1: gt_title (ceiling) ---
    result_gt = _evaluate_mode(
        dataset, kb, args.lookup_limit, args.rrf_k,
        name_override=None, mode_label="gt_title", desc="gt_title mode",
    )
    print()
    _print_mode_report(result_gt)
    results_to_save = [result_gt]

    # --- Mode 2: predicted names (realistic) ---
    if args.names_file:
        if not os.path.exists(args.names_file):
            print(f"WARNING: --names-file not found: {args.names_file}  (skipping)")
        else:
            predicted = load_predicted_names(args.names_file, args.names_variant)
            matched = sum(1 for it in dataset if it["unique_id"] in predicted)
            print(f"\nLoaded {len(predicted)} predicted names "
                  f"(variant={args.names_variant!r}); "
                  f"{matched}/{len(dataset)} dataset examples matched.\n")

            result_pred = _evaluate_mode(
                dataset, kb, args.lookup_limit, args.rrf_k,
                name_override=predicted, mode_label="predicted",
                desc="predicted mode",
            )
            print()
            _print_mode_report(result_pred)
            results_to_save.append(result_pred)

            # Side-by-side summary
            print("\n-- Side-by-side summary (recall@1) " + "-" * 33)
            print(f"{'strategy':14s}  {'gt_title':>10s}  {'predicted':>10s}  {'delta':>8s}")
            print("-" * 48)
            for s in STRATEGIES:
                gt_r1  = result_gt["recall"][s]["1"]
                pr_r1  = result_pred["recall"][s]["1"]
                marker = "  <- current" if s == "category" else ""
                print(f"{s:14s}  {100*gt_r1:9.1f}%  {100*pr_r1:9.1f}%  "
                      f"{100*(pr_r1-gt_r1):+7.1f}%{marker}")
            print("=" * 70)

    for r in results_to_save:
        r.pop("_hits_raw", None)
    out = {"lookup_limit": args.lookup_limit, "rrf_k": args.rrf_k,
           "modes": results_to_save}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nPer-example detail: {args.output}")


if __name__ == "__main__":
    main()
