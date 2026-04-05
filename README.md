# Sydney AI Research Lab

Local-first research lab for eval-driven iteration over open models.

This repo is intentionally small and boring:

- ingest local docs and notes
- chunk them with metadata
- build a lightweight local retrieval index
- run frozen eval tasks against variants
- log results to SQLite and JSON artifacts
- derive synthetic training examples from good outputs
- offer optional Ollama or llama.cpp inference backends
- prepare real PEFT / LoRA training jobs from synthetic outputs

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
- `ai_lab/model_adapters.py` - local model backends for Ollama, llama.cpp, or fallback heuristic
- `ai_lab/storage.py` - SQLite-backed run/result logging
- `config/` - sample configs
- `storage/` - sample docs, datasets, and runtime artifacts

## What Works In The MVP

- ingest docs from `storage/docs`
- chunk them into stable IDs
- cache chunk/index artifacts by document fingerprint
- build a TF-IDF-like sparse retrieval index
- query the index from the CLI
- run baseline and RAG-style experiments through a local model adapter
- score answers with task-specific rubrics and pass/fail checks
- save run artifacts and results to `storage/lab.sqlite3`
- derive synthetic JSONL from better-scoring outputs
- show the latest run and comparison history in the local dashboard
- prepare a PEFT/LoRA training plan from synthetic outputs

## Run It Locally

From the repo root:

```bash
python3 -m ai_lab.cli run --config config/lab.sample.yaml
python3 -m ai_lab.cli query "What is the recommended research cadence?" --config config/lab.sample.yaml
python3 -m ai_lab.cli serve --config config/lab.sample.yaml
python3 -m ai_lab.cli train --config config/lab.sample.yaml --synthetic storage/synthetic/<latest-run>.jsonl --dry-run
```

Then open `http://127.0.0.1:8787`.

## Optional Backends

- Set `model.provider: ollama` in [`config/lab.sample.yaml`](/Users/sydneyjackson/ai-lab/config/lab.sample.yaml) if you have Ollama running locally.
- Set `model.provider: llamacpp` and point `model.llama_command` at your local llama.cpp binary or wrapper command.
- Leave `model.provider: auto` to try Ollama, then llama.cpp, then the heuristic fallback.

## Training Path

The PEFT entrypoint is [`ai_lab/scripts/train_peft_sft.py`](/Users/sydneyjackson/ai-lab/ai_lab/scripts/train_peft_sft.py).

It currently:

- reads the synthetic JSONL produced by experiments
- writes a PEFT training manifest and JSON config
- supports `--dry-run` for wiring validation
- runs a real `transformers` + `peft` SFT job when the ML dependencies are installed

## Suggested Next Steps

1. Add a richer prompt library for different workflow types.
2. Expand the dashboard with per-task drill-down and answer diffs.
3. Add incremental document watching so caches update automatically.
4. Add a benchmark pack for latency and token-cost tracking.
5. Add cloud training support that reuses the same synthetic dataset format.
