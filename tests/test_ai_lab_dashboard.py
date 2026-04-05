import json
import tempfile
import unittest
from pathlib import Path

from ai_lab.dashboard import build_lab_state, render_dashboard_html


class BuildLabStateTests(unittest.TestCase):
    def test_build_lab_state_reads_config_manifest_and_datasets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config.yaml").write_text(
                "project_name: sydney-ai-lab\n"
                "base_model: Qwen/Qwen3-1.7B\n"
                "hardware_profile: apple-silicon-16gb\n"
                "train_file: data/train.jsonl\n"
                "eval_file: data/eval.jsonl\n"
                "output_dir: runs/qwen3-1.7b-lora\n"
            )
            (root / "data").mkdir()
            (root / "data" / "train.jsonl").write_text('{"prompt":"p1","completion":"c1"}\n{"prompt":"p2","completion":"c2"}\n')
            (root / "data" / "eval.jsonl").write_text('{"prompt":"ep1","completion":"ec1"}\n')
            (root / "runs").mkdir()
            (root / "runs" / "manifest.json").write_text(json.dumps({"created_at": "2026-03-21T22:15:00Z"}))

            state = build_lab_state(root / "config.yaml", root / "runs" / "manifest.json")

            self.assertEqual(state["project_name"], "sydney-ai-lab")
            self.assertEqual(state["base_model"], "Qwen/Qwen3-1.7B")
            self.assertEqual(state["dataset_counts"]["train"], 2)
            self.assertEqual(state["dataset_counts"]["eval"], 1)
            self.assertEqual(state["last_run"]["created_at"], "2026-03-21T22:15:00Z")


class RenderDashboardHtmlTests(unittest.TestCase):
    def test_render_dashboard_html_contains_core_sections(self):
        html = render_dashboard_html(
            {
                "project_name": "sydney-ai-lab",
                "base_model": "Qwen/Qwen3-1.7B",
                "hardware_profile": "apple-silicon-16gb",
                "dataset_counts": {"train": 2, "eval": 1},
                "last_run": {"created_at": "2026-03-21T22:15:00Z"},
                "status": "ready",
                "train_file": "data/train.jsonl",
                "eval_file": "data/eval.jsonl",
                "output_dir": "runs/qwen3-1.7b-lora",
            }
        )

        self.assertIn("Sydney AI Research Lab", html)
        self.assertIn("Qwen/Qwen3-1.7B", html)
        self.assertIn("apple-silicon-16gb", html)
        self.assertIn("Train samples", html)
        self.assertIn("2", html)
        self.assertIn("ready", html)


if __name__ == "__main__":
    unittest.main()
