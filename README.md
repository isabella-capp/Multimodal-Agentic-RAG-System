# Encyclopedic-VQA — Agentic RAG (Phase C)

Agentic retrieval-augmented VQA on Encyclopedic-VQA. A vision-language model,
served on **vLLM** or reached over any OpenAI-compatible endpoint, drives a
tool-using loop over a visual retrieval stack (EVA-CLIP + FAISS), a SQLite
knowledge base, and a cross-encoder reranker. Three tools, and the agent grows
its own working set of articles:

1. **`lookup_article(name)`** — resolve the entity the model recognises in the
   image to its Wikipedia article, by name. This is the only channel that can
   beat the ~47% recall ceiling of the image embedding.
2. **`search_by_image()`** — list the articles whose reference images match the
   query image; the fallback when the model cannot name what it sees.
3. **`read_article(title, query)`** — cross-encoder over *all* the paragraphs of
   a chosen article.

The model may use its own knowledge to **name** what it sees (that is only a
search key), never to **answer**: every fact must come from a retrieved passage.

## Requirements

- SLURM cluster with the `cvcs2026` account and an A40/L40S (45 GB) GPU.
- [`uv`](https://docs.astral.sh/uv/) on `PATH` (project + scoring envs are synced
  automatically on first run).
- Shared assets already present under `/work/cvcs2026/` for the account:
  - HuggingFace cache (`recursive_retrievers/hf_cache`) with Qwen2.5-VL-3B/7B,
    EVA-CLIP-8B, `bge-reranker-base`, `clip-vit-large-patch14`.
  - `encyclopedic/`: `knn.index`, `knn.json`, `encyclopedic_kb_wiki.db`,
    `encyclopedic_test_subset.json`.
- Clone the repo to your home (e.g. `/homes/$USER/cvcs2026`) and submit from its root.

## Knowledge base and name index (SQLite)

`encyclopedic_kb_wiki.db` (~20 GB) holds the ~2.0M articles in `articles`
(url, title) and their text in `paragraphs`, plus two derived tables that turn an
entity *name* into an article:

- **`aliases`** (`alias`, `url`) — normalised surface forms of every title
  (accents and punctuation stripped, `(disambiguation)` suffix dropped, leading
  `the` removed), indexed for exact lookup.
- **`titles_fts`** — an FTS5 index over the title tokens, used only as a fuzzy
  fallback (BM25 shortlist, re-scored by token overlap).

Matching is pure string matching, deliberately **not** embeddings: the EVA-CLIP
text tower is misaligned with the image index (0% recall@50 even when given the
ground-truth title), and exact matching is anyway more precise than similarity
for near-identical names. Ceiling on the test subset: **83.5%** of articles are
reachable by name, against **46.7%** for the image embedding.

The article set never changes, so the tables are precomputed once. A full build
produces them automatically; `--index-only` rebuilds just them on an existing KB
(seconds, instead of re-ingesting the 15 GB source JSON):

```bash
sbatch scripts/build_kb_sqlite.sh                 # full KB, name index included
sbatch scripts/build_kb_sqlite.sh --index-only    # only the name tables
```

Use it through `KnowledgeBase`, the single point of access to the DB:

```python
kb = KnowledgeBase("/work/cvcs2026/encyclopedic/encyclopedic_kb_wiki.db")
kb.lookup_articles("Northern cardinal")   # [{'title': ..., 'wiki_url': ..., 'match': 'exact'}]
kb.get_paragraphs_by_url(wiki_url=...)    # the article text
```

Normalisation at build time and at query time must stay identical — a mismatch
fails silently — so both use the helpers in `src/retrieval/knowledge_base.py`.

## Run

```bash
export LLM_API_KEY=sk-or-v1-...                       # only for remote models
sbatch --export=ALL scripts/run_model_sweep.sh        # 3B, 7B and a remote model
sbatch --export=ALL scripts/run_smoke.sh              # 5 examples, checks a remote model works
sbatch --export=ALL scripts/run_naming_probe.sh       # naming ability only, no GPU
```

The sweep serves each local model on vLLM (staging the weights to node-local disk
so loading is fast and immune to `/work` contention), runs a naming probe and the
full evaluation, and scores the predictions. It skips any model already scored, so
re-submitting only fills the gaps. The first run also builds an isolated vLLM venv
at `/homes/$USER/vllm_venv`.

## Outputs

Under `outputs/agentic/sweep/` (git-ignored), per model tag:

- `naming_<tag>.jsonl` — predicted entity name, candidates, whether it resolved
- `predictions_<tag>.jsonl` — predictions + the agent trace for each example
- `predictions_<tag>.metrics.json` — tool usage, miss rate per tool, entry tool,
  call sequences, timing
- `results_<tag>.json` — Encyclopedic-VQA accuracy (EM + BEM)

Progress, the naming-probe summary, and a few full agent traces go to
`logs/sweep_<jobid>.err`.
