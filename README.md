# Sydney AI Research Lab

Local-first research lab for eval-driven iteration over open models.

This repo is intentionally small and boring:

- ingest local docs and notes
- chunk them with metadata
- build a lightweight local retrieval index
- run frozen eval tasks against variants
- log results to SQLite and JSON artifacts
- derive synthetic training examples from good outputs
- leave clear stubs for LoRA and QLoRA training later

## Architecture

- `ai_lab/app.py` - simple HTML dashboard served locally
- `ai_lab/cli.py` - command line entrypoint for run/query/serve
- `ai_lab/pipeline/ingest.py` - load markdown, text, json, and jsonl docs
- `ai_lab/pipeline/chunking.py` - split documents into retrievable chunks
- `ai_lab/pipeline/retrieval.py` - sparse local embedding/index/search
- `ai_lab/pipeline/evals.py` - eval task loading and scoring
- `ai_lab/pipeline/synthetic.py` - generate instruction-response examples from strong runs
- `ai_lab/pipeline/experiments.py` - orchestrate the eval-driven experiment loop
- `ai_lab/pipeline/training.py` - prepare LoRA datasets and manifests
- `ai_lab/storage.py` - SQLite-backed run/result logging
- `config/` - sample configs
- `storage/` - sample docs, datasets, and runtime artifacts

## What Works In The MVP

- ingest docs from `storage/docs`
- chunk them into stable IDs
- build a TF-IDF-like sparse retrieval index
- query the index from the CLI
- run baseline and RAG-style experiments
- score answers automatically against eval tasks
- save run artifacts and results to `storage/lab.sqlite3`
- derive synthetic JSONL from better-scoring outputs
- show the latest run in the local dashboard

## Run It Locally

From the repo root:

```bash
python -m ai_lab.cli run --config config/lab.sample.yaml
python -m ai_lab.cli query "What is the recommended research cadence?" --config config/lab.sample.yaml
python -m ai_lab.cli serve --config config/lab.sample.yaml
```

Then open `http://127.0.0.1:8787`.

## Suggested Next Steps

1. Swap the heuristic answer generator for a real local model backend, starting with Ollama or llama.cpp.
2. Add a persistent chunk/index cache so repeated runs are faster.
3. Define a richer eval rubric with pass/fail and numeric grading.
4. Add a small task browser in the dashboard.
5. Wire LoRA dataset export into a real PEFT training script.

