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
                        "delta": 0.0434,
                        "synthetic_examples": 1,
                    }
                ],
            }
        )

        self.assertIn("Run Comparison", html)
        self.assertIn("comparison-table", html)
        self.assertIn("heuristic:local-rule-based", html)
        self.assertIn("hit", html)


if __name__ == "__main__":
    unittest.main()
