import unittest

from ai_lab.dashboard import render_dashboard_html


class DashboardComparisonTests(unittest.TestCase):
    def test_render_dashboard_html_includes_comparison_table(self) -> None:
        html = render_dashboard_html(
            {
                "project_name": "sydney-ai-lab",
                "base_model": "Qwen/Qwen3-1.7B",
                "hardware_profile": "apple-silicon-16gb",
                "dataset_counts": {"train": 3, "eval": 3},
                "last_run": {"created_at": "2026-03-21T22:15:00Z"},
                "status": "ready",
                "train_file": "storage/datasets/train.jsonl",
                "eval_file": "storage/datasets/evals.jsonl",
                "output_dir": "storage/runs",
                "model_backend": "heuristic:local-rule-based",
                "cache_state": "hit",
                "loops": [],
                "reports": [],
                "experiments": [],
                "comparisons": [
                    {
                        "created_at": "2026-04-05T14:51:21Z",
                        "name": "sydney-ai-research-lab",
                        "docs": 2,
                        "chunks": 3,
                        "baseline_avg": 0.081,
                        "rag_avg": 0.1244,
                        "pass_rate": 0.33,
                        "avg_latency_seconds": 0.27,
                        "estimated_cost_usd": 0.0,
                        "delta": 0.0434,
                        "synthetic_examples": 1,
                    }
                ],
                "task_diffs": [
                    {
                        "task_id": "lab-products",
                        "delta": 0.114,
                        "baseline": {"answer": "Baseline answer", "score": 0.11, "passed": False, "metrics": {"latency_seconds": 0.31, "estimated_cost_usd": 0.0}},
                        "rag": {"answer": "RAG answer", "score": 0.224, "passed": True, "metrics": {"latency_seconds": 0.26, "estimated_cost_usd": 0.0}, "retrieved_chunks": [{"doc_id": "doc-1", "preview": "Useful evidence"}]},
                        "baseline_score": 0.11,
                        "rag_score": 0.224,
                        "baseline_passed": False,
                        "rag_passed": True,
                        "baseline_latency_seconds": 0.31,
                        "rag_latency_seconds": 0.26,
                        "baseline_cost_usd": 0.0,
                        "rag_cost_usd": 0.0,
                        "baseline_prompt_tokens": 11,
                        "rag_prompt_tokens": 12,
                        "baseline_completion_tokens": 5,
                        "rag_completion_tokens": 7,
                        "retrieved_chunks": [{"doc_id": "doc-1", "preview": "Useful evidence"}],
                    }
                ],
            }
        )

        self.assertIn("Run Comparison", html)
        self.assertIn("comparison-table", html)
        self.assertIn("heuristic:local-rule-based", html)
        self.assertIn("hit", html)
        self.assertIn("Task Diff View", html)
        self.assertIn("diff-answer", html)


if __name__ == "__main__":
    unittest.main()
