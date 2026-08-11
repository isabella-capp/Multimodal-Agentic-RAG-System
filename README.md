# Encyclopedic-VQA — Agentic RAG (Phase C)

Agentic retrieval-augmented VQA on Encyclopedic-VQA. A Qwen2.5-VL model, served
on **vLLM**, drives a tool-using loop over a visual retrieval stack
(EVA-CLIP + FAISS) and a cross-encoder reranker:

1. **`research`** (forced first) — retrieve the top articles for the image, then a
   multimodal *extractor* sub-agent picks the right article and reports the key
   evidence.
2. **`search_paragraphs`** (optional) — cross-encoder refine over the same article
   pool for a second hop when evidence is missing.
3. The main model answers **only** from the gathered evidence.

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

## Run

```bash
sbatch scripts/run_agentic_vllm.sh        # 3B (default)
sbatch scripts/run_agentic_vllm.sh 7b     # 7B
```

The job serves vLLM (staging the weights to node-local disk so loading is fast and
immune to `/work` contention), runs the eval, and scores the predictions. The
first run also builds an isolated vLLM venv at `/homes/$USER/vllm_venv`.

## Outputs

Under `outputs/agentic/` (git-ignored):

- `predictions_agentic_<size>_research.jsonl` — per-example predictions + agent trace
- `predictions_agentic_<size>_research.metrics.json` — tool-use / timing metrics
- `results_agentic_<size>_research.json` — Encyclopedic-VQA accuracy (EM + BEM)

Progress and one full agent trace are logged to `logs/agentic_vllm_<jobid>.err`.
