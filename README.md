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

Three settings are compared, always at a fixed model:

| | |
|---|---|
| **A** | no retrieval — what the model answers on its own |
| **B** | retrieval + cross-encoder, the standard RAG baseline |
| **C** | the agentic loop above |

## Requirements

- SLURM cluster with the `cvcs2026` account and an A40/L40S (45 GB) GPU.
- [`uv`](https://docs.astral.sh/uv/) on `PATH` (project + scoring envs are synced
  automatically on first run).
- Shared assets already present under `/work/cvcs2026/` for the account:
  - HuggingFace cache (`recursive_retrievers/hf_cache`) with Qwen2.5-VL-3B/7B,
    Qwen3-VL-8B, EVA-CLIP-8B, `bge-reranker-base`, `clip-vit-large-patch14`
    (add more with `scripts/setup/download_model.sh <hf-id>`).
  - `encyclopedic/`: `knn.index`, `knn.json`, `encyclopedic_kb_wiki.db`,
    `encyclopedic_test_subset.json`.
- Clone the repo to your home (e.g. `/homes/$USER/cvcs2026`) and submit from its root.

## Running experiments

**Always launch through `scripts/submit.sh`.** It snapshots the code before
submitting, which is what makes a queued job run what you actually submitted:

```bash
scripts/submit.sh scripts/run_abc.sh                       # A, B and C, one model
VARIANT=lookup-first scripts/submit.sh scripts/run_abc.sh  # a named attempt
SMOKE=1 scripts/submit.sh scripts/run_abc.sh --time=00:40:00   # 5 examples

export LLM_API_KEY=sk-or-v1-...                            # only for remote models
scripts/submit.sh scripts/agentic/run_sweep.sh             # C across model sizes
scripts/submit.sh scripts/agentic/run_smoke.sh             # does a remote model work at all
```

Anything after the script path is passed through to `sbatch`.

`run_abc.sh` runs the three settings against **one vLLM server in one job**, so
they differ only in method — same weights, same endpoint, same prompt format,
same examples. `run_sweep.sh` answers the other question, C across model sizes,
and skips any model already scored so re-submitting only fills the gaps.

Jobs serve their model on a port derived from the SLURM job id and verify
`/v1/models` before running: `localhost` is per node, and with a fixed port a
colleague's vLLM on the same node silently answers your requests — which once
cost a full run of empty predictions.

Use `SMOKE=1` when switching model. Bad image preprocessing degrades answers
without raising anything, and five examples show it immediately.

## Working on variants

`sbatch` copies the `.sh` at submission but reads the `.py` **when the job
starts**. Editing a strategy while an earlier job is queued therefore changes
what that job runs, and loses the version you meant to test. `submit.sh` avoids
both by copying `src/`, `scripts/` and the scorer into `runs/<id>/` and pointing
the job there — 460 KB per run, and the working tree is yours again the moment
the job is submitted.

```
runs/20260815-104401-lookup-first/
├── src/  scripts/  evqa_eval/   the exact code that ran
├── RUN_INFO                     run id, variant, commit, branch, dirty count
├── uncommitted.diff             changes not in git at submit time
└── JOB_ID
```

So the loop is: edit → `VARIANT=name scripts/submit.sh …` → edit again for the
next idea, without waiting. Each run keeps its own code, its own
`logs/<run-id>/` and its own `outputs/abc/<model>/<run-id>/`, so attempts never
overwrite each other.

Every predictions file also gets a `.meta.json` recording the run id, variant,
commit, dirty flag and the exact command. Under `submit.sh` those come from the
snapshot's `RUN_INFO`, captured at **submit** time — reading git when the job
starts would report whatever the tree holds by then, which is exactly what
changes while a job waits in the queue.

Git branches are still the right tool once a variant *wins* and you want to keep
it; `runs/` tells you which one that was.

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
sbatch scripts/setup/build_kb_sqlite.sh                 # full KB, name index included
sbatch scripts/setup/build_kb_sqlite.sh --index-only    # only the name tables
```

Use it through `KnowledgeBase`, the single point of access to the DB:

```python
kb = KnowledgeBase("/work/cvcs2026/encyclopedic/encyclopedic_kb_wiki.db")
kb.lookup_articles("Northern cardinal")   # [{'title': ..., 'wiki_url': ..., 'match': 'exact'}]
kb.get_paragraphs_by_url(wiki_url=...)    # the article text
```

Normalisation at build time and at query time must stay identical — a mismatch
fails silently — so both use the helpers in `src/retrieval/knowledge_base.py`.

## Layout

```
src/
  paths.py prompts.py llm.py runner.py provenance.py   shared by every setting
  vlm/        arg_parser  dataset  run_inference       A and B
  agent/      prompts messages tools rag run metrics   C
              run_inference  run_naming_probe
  retrieval/  retriever  knowledge_base  reranker  build_kb_sqlite
scripts/
  submit.sh          snapshot + submit — the way to launch
  run_abc.sh         A, B and C for one model
  lib/vllm.sh        serving lifecycle shared by every experiment
  agentic/  baselines/  retrieval/  setup/
```

`runner.run_batch` owns the loop, the thread pool, the resume and the writing;
each script supplies only its `predict`. `llm.chat_model` is the only place a
chat client is built. Keeping those single means A, B and C cannot drift apart
without someone noticing.

## Outputs

Under `outputs/` (git-ignored):

- `abc/<model>/<run-id>/` — `predictions_A|B|C.jsonl`, `results_A|B|C.json`, and
  a `.meta.json` per prediction file
- `agentic/sweep/` — the model sweep, per tag: `naming_<tag>.jsonl` (predicted
  entity name and whether it resolved), `predictions_<tag>.jsonl`,
  `predictions_<tag>.metrics.json` (tool usage, miss rate per tool, entry tool,
  call sequences) and `results_<tag>.json`
- `final_test/`, `ablation*/`, `retrieval/` — the phase-B record, kept as-is
- `_archive/` — results from pipelines that no longer exist

Logs go to `logs/<run-id>/`, one directory per run: the SLURM `.out`/`.err` and
the vLLM server log together.
