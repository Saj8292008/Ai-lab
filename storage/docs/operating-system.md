# AI Research Lab Operating System

## Mission
Turn local open models into specialized, defensible startup assets.

## Research doctrine
- inspect everything before scaling
- start with the smallest possible experiment
- isolate failures fast with tiny datasets and narrow evals
- write down exactly what changed between runs

## Lab lanes
### Data lane
Own:
- dataset collection
- deduplication
- schema checks
- licensing notes
- train/eval split

Required artifact per dataset:
- source note
- intended use
- forbidden use
- sample inspection notes

### Training lane
Default order:
- prompt engineering baseline
- LoRA baseline
- QLoRA on stronger hardware
- full finetune only if adapters plateau

### Eval lane
Every experiment must answer:
- did it beat the base model?
- on what exact prompts?
- by how much?
- what new failure mode appeared?

Minimum eval bundle:
- 20 hand-picked prompts
- pass/fail rubric
- latency note
- hallucination note

### Product lane
A run becomes a product candidate if it is:
- measurably better than base
- cheap enough to run locally
- useful for a real niche user
- maintainable by a small team

## Hardware strategy
### Apple Silicon / smaller local box
Use for:
- data cleaning
- tiny LoRA experiments
- evals
- packaging

### Larger NVIDIA box
Use for:
- 7B to 14B QLoRA work
- batched evals
- generation benchmarking

## Research cadence
Daily:
- inspect data
- run one controlled experiment
- log one finding

Weekly:
- compare winning runs
- kill weak directions
- package one demo or case study

Monthly:
- pick one niche worth selling into
- create one customer-facing offer
- publish one technical insight

## Startup scoreboard
Track:
- datasets created
- eval win rate vs base model
- average training time
- average inference cost
- number of reusable LoRAs
- paid pilots or demos created

