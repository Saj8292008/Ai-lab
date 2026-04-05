import json
import tempfile
import unittest
from pathlib import Path

from ai_lab.dashboard import build_lab_state, render_dashboard_html


class PhaseOneDashboardTests(unittest.TestCase):
    def test_build_lab_state_includes_default_loops_and_recent_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "ai_lab" / "configs").mkdir(parents=True)
            (root / "ai_lab" / "configs" / "config.yaml").write_text(
                "project_name: sydney-ai-lab\n"
                "base_model: Qwen/Qwen3-1.7B\n"
                "hardware_profile: apple-silicon-16gb\n"
                "train_file: data/train.jsonl\n"
                "eval_file: data/eval.jsonl\n"
                "output_dir: runs/qwen3-1.7b-lora\n"
            )
            (root / "data").mkdir()
            (root / "data" / "train.jsonl").write_text('{"prompt":"p1","completion":"c1"}\n')
            (root / "data" / "eval.jsonl").write_text('{"prompt":"ep1","completion":"ec1"}\n')
            (root / "runs").mkdir()
            (root / "runs" / "manifest.json").write_text(json.dumps({"created_at": "2026-03-21T22:15:00Z"}))
            (root / "reports").mkdir()
            (root / "reports" / "2026-03-22T07-30-00Z--rd-council.json").write_text(
                json.dumps(
                    {
                        "loop_id": "rd-council",
                        "title": "R&D Council",
                        "summary": "Council recommends benchmark automation.",
                        "findings": ["Need frozen evals"],
                        "next_actions": ["Add benchmark runner"],
                        "created_at": "2026-03-22T07:30:00Z",
                    }
                )
            )

            state = build_lab_state(
                root / "ai_lab" / "configs" / "config.yaml",
                root / "runs" / "manifest.json",
            )

            self.assertEqual(len(state["loops"]), 3)
            self.assertEqual(state["loops"][0]["id"], "prompt-optimizer")
            self.assertEqual(len(state["reports"]), 1)
            self.assertEqual(state["reports"][0]["title"], "R&D Council")

    def test_render_dashboard_html_contains_loops_reports_and_cleaner_shell_layout(self):
        html = render_dashboard_html(
            {
                "project_name": "sydney-ai-lab",
                "base_model": "Qwen/Qwen3-1.7B",
                "hardware_profile": "apple-silicon-16gb",
                "dataset_counts": {"train": 3, "eval": 2},
                "last_run": {"created_at": "2026-03-21T22:15:00Z"},
                "status": "ready",
                "train_file": "data/train.jsonl",
                "eval_file": "data/eval.jsonl",
                "output_dir": "runs/qwen3-1.7b-lora",
                "loops": [
                    {"name": "Prompt optimization", "cadence": "every 5m", "status": "active", "goal": "Improve prompts"}
                ],
                "reports": [
                    {"title": "R&D Council", "summary": "Council recommends eval automation.", "created_at": "2026-03-22T07:30:00Z"}
                ],
            }
        )

        self.assertIn("lab-shell", html)
        self.assertIn("tab-button", html)
        self.assertIn("Overview", html)
        self.assertIn("Experiments", html)
        self.assertIn("Reports", html)
        self.assertIn("experiment-chart", html)
        self.assertIn("line-success", html)
        self.assertIn("line-failure", html)
        self.assertIn("Prompt optimization", html)
        self.assertIn("Council recommends eval automation.", html)
        self.assertIn("Report summary", html)


if __name__ == "__main__":
    unittest.main()
