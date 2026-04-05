import json
import tempfile
import unittest
from pathlib import Path

from ai_lab.pipeline.experiments import run_experiment
from ai_lab.storage import LabStore


class ExperimentRunnerTests(unittest.TestCase):
    def test_run_experiment_writes_artifacts_and_prefers_rag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = root / "storage"
            docs_dir = storage / "docs"
            datasets_dir = storage / "datasets"
            docs_dir.mkdir(parents=True)
            datasets_dir.mkdir(parents=True)

            (docs_dir / "operating-system.md").write_text(
                "# AI Research Lab Operating System\n\n"
                "The first three lab products to test are a private business assistant LoRA, a niche creator style/model pack, and a local knowledge worker copilot for docs and drafting.\n\n"
                "Apple Silicon is for data cleaning, tiny LoRA experiments, evals, and packaging. Larger NVIDIA hardware is for 7B to 14B QLoRA work, batched evals, and generation benchmarking.\n"
            )
            (datasets_dir / "evals.jsonl").write_text(
                '{"task_id":"lab-products","prompt":"What are the first three lab products to test?","reference_answer":"The first three lab products to test are a private business assistant LoRA, a niche creator style/model pack, and a local knowledge worker copilot for docs and drafting.","expected_terms":["private business assistant lora","niche creator style model pack","local knowledge worker copilot"],"source_doc_ids":[]}\n'
                '{"task_id":"hardware-strategy","prompt":"What is the recommended hardware strategy for Apple Silicon and a larger NVIDIA box?","reference_answer":"Apple Silicon is for data cleaning, tiny LoRA experiments, evals, and packaging. Larger NVIDIA hardware is for 7B to 14B QLoRA work, batched evals, and generation benchmarking.","expected_terms":["apple silicon","tiny lora experiments","7b","14b","qlora"],"source_doc_ids":[]}\n'
            )

            config = {
                "project_name": "sydney-ai-research-lab",
                "docs_dir": str(docs_dir),
                "eval_file": str(datasets_dir / "evals.jsonl"),
                "chunk_words": 60,
                "chunk_overlap": 10,
                "top_k": 4,
                "min_synthetic_score": 0.1,
            }

            summary = run_experiment(config=config, store=LabStore(storage))

            self.assertIn("run_id", summary)
            self.assertIn("metrics", summary)
            self.assertTrue((storage / "runs").exists())
            self.assertTrue((storage / "eval_results").exists())

            run_record = LabStore(storage).latest_run()
            self.assertIsNotNone(run_record)
            metrics = json.loads(run_record["metrics_json"])
            self.assertGreater(metrics["variants"]["rag"]["avg_score"], metrics["variants"]["baseline"]["avg_score"])


if __name__ == "__main__":
    unittest.main()

