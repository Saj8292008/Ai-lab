# ai_lab Package Notes

This package holds the lab runtime, pipeline, and dashboard code.

For the current MVP, start with the repo root [`README.md`](/Users/sydneyjackson/ai-lab/README.md) and run the CLI from there:

```bash
python3 -m ai_lab.cli run --config config/lab.sample.yaml
```

The older phase-one helper scripts are still present for compatibility, but the new eval-driven loop lives in `ai_lab/cli.py` and `ai_lab/pipeline/`.

