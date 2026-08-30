# Sydney AI Research Lab

A local-first experimentation environment for studying **AI evaluation, retrieval-augmented generation (RAG), model comparison, and iterative improvement**.

I built this project as a hands-on way to learn an important question in AI development:

> **How can we tell whether a change to an AI system actually makes it better?**

Instead of relying only on subjective impressions, the lab runs repeatable evaluation tasks, compares system variants, records results, and preserves experiment artifacts for later analysis.

## Why I Built This

As I started building with AI systems, I became interested in the difference between making a model *look* better and measuring whether it is actually more useful or reliable.

This project is my environment for learning that process through experimentation. It lets me test changes such as retrieval, prompts, model backends, and synthetic training data against a consistent evaluation set.

The project is intentionally local-first so experiments can be understandable, reproducible, and inexpensive to run.

## Research Loop

```text
Documents / Notes
       ↓
Ingestion + Chunking
       ↓
Retrieval Index
       ↓
Baseline ─────────────┐
       ↓              │
RAG / Variant         │
       ↓              │
Evaluation Tasks      │
       ↓              │
Scores + Comparisons ←┘
       ↓
Experiment Logs
       ↓
Synthetic Data / Next Experiment
```

The goal is to make improvement **measurable rather than assumed**.

## What the Lab Can Do

### Evaluation

- Run frozen evaluation tasks across multiple system variants
- Score outputs with task-specific rubrics and pass/fail checks
- Compare baseline and RAG results task by task
- Preserve experiment results for repeatable comparison

### Retrieval

- Ingest Markdown, text, JSON, and JSONL documents
- Split documents into stable, metadata-aware chunks
- Build a lightweight TF-IDF-style local retrieval index
- Cache retrieval artifacts using document fingerprints
- Query the index from the command line

### Experiment Tracking

- Record experiment results in SQLite and JSON artifacts
- Track latency, prompt tokens, completion tokens, and estimated cost
- Display the latest experiment and comparison history in a local dashboard
- Automatically refresh changed documents and optionally rerun experiments

### Model Experimentation

- Run through local Ollama models
- Use llama.cpp-compatible backends
- Fall back to a simple heuristic adapter for pipeline testing
- Compare baseline and retrieval-augmented configurations

### Synthetic Data & Training

- Derive synthetic instruction/response examples from stronger experiment outputs
- Export synthetic datasets as JSONL
- Prepare PEFT / LoRA training manifests
- Validate training configuration with a dry-run workflow
- Generate an SSH-ready cloud training bundle

## Architecture

```text
ai_lab/
├── app.py                 # local experiment dashboard
├── cli.py                 # run, query, serve, train, and watch commands
├── model_adapters.py      # Ollama, llama.cpp, and fallback adapters
├── storage.py             # SQLite-backed experiment logging
├── pipeline/
│   ├── ingest.py          # document ingestion
│   ├── chunking.py        # metadata-aware document chunking
│   ├── retrieval.py       # local sparse retrieval/indexing
│   ├── evals.py           # evaluation loading and scoring
│   ├── prompts.py         # baseline, RAG, and synthesis prompts
│   ├── experiments.py     # experiment orchestration
│   ├── synthetic.py       # synthetic example generation
│   ├── training.py        # LoRA dataset / manifest preparation
│   ├── watching.py        # document refresh + experiment watcher
│   └── cloud.py           # remote training bundle generation
└── scripts/
    └── train_peft_sft.py  # PEFT supervised fine-tuning entrypoint

config/                    # example experiment configuration
storage/                   # sample documents, datasets, and run artifacts
```

## Quick Start

From the repository root:

```bash
python3 -m ai_lab.cli run --config config/lab.sample.yaml
```

Query the retrieval index:

```bash
python3 -m ai_lab.cli query "What is the recommended research cadence?" \
  --config config/lab.sample.yaml
```

Launch the local dashboard:

```bash
python3 -m ai_lab.cli serve --config config/lab.sample.yaml
```

Then open:

```text
http://127.0.0.1:8787
```

Run the document watcher once:

```bash
python3 -m ai_lab.cli watch --config config/lab.sample.yaml --once
```

Prepare a training run without launching training:

```bash
python3 -m ai_lab.cli train \
  --config config/lab.sample.yaml \
  --synthetic storage/synthetic/<latest-run>.jsonl \
  --dry-run
```

## Model Backends

The sample configuration supports three modes:

- **Ollama** — local model inference
- **llama.cpp** — local inference through a compatible command or wrapper
- **Auto** — tries Ollama, then llama.cpp, then the heuristic fallback

See [`config/lab.sample.yaml`](config/lab.sample.yaml) for configuration.

## Training Path

The PEFT training entrypoint is [`ai_lab/scripts/train_peft_sft.py`](ai_lab/scripts/train_peft_sft.py).

It can:

- read synthetic JSONL generated from experiments
- create a PEFT training manifest and JSON configuration
- perform dry-run validation
- launch a `transformers` + `peft` supervised fine-tuning job when ML dependencies are installed
- generate a cloud-ready training bundle when cloud mode is enabled

## What I'm Evaluating

This project is evolving toward questions such as:

- When does retrieval improve an answer, and when does it make it worse?
- How should AI-system changes be evaluated beyond a few hand-picked examples?
- How stable are results across repeated tasks and prompt variants?
- Can failure cases be turned into useful future evaluation examples?
- How should latency and cost be balanced against quality improvements?
- What makes an evaluation meaningful rather than merely easy to pass?

## Current Limitations

This is an experimental learning project, not a production ML platform.

Current limitations include:

- the retrieval system is intentionally lightweight rather than embedding-model based
- evaluation rubrics are still small and need broader coverage
- the fallback model adapter is for pipeline validation, not meaningful model benchmarking
- cloud training currently prepares bundles rather than managing a complete remote training lifecycle
- synthetic examples should be reviewed carefully before being treated as high-quality training data

Documenting these limitations is part of the project: I want the repository to show not only what works, but also what still needs to be tested.

## Next Experiments

1. Expand the evaluation suite with more diverse and adversarial tasks.
2. Add model-based judges while comparing them against deterministic scoring.
3. Add hosted-model adapters for controlled cross-model comparisons.
4. Build a replay view for comparing prompts, retrieved context, and outputs side by side.
5. Add dataset-quality checks before synthetic examples enter a training set.
6. Track repeated-run variance instead of relying on single measurements.
7. Add failure categorization for hallucination, retrieval error, instruction failure, and incomplete answers.

## What I'm Learning

This project is helping me develop practical understanding of:

`Python` · `AI Evaluation` · `RAG` · `Experiment Design` · `Retrieval` · `SQLite` · `Local Models` · `Synthetic Data` · `PEFT / LoRA` · `Model Reliability`

I use AI-assisted development tools as part of my workflow, but my goal is to understand the systems I build: how the pieces interact, how to test them, where they fail, and how to improve them through evidence rather than intuition.

---

**Status:** Active learning / research project
