import json
import tempfile
import unittest
from pathlib import Path

from ai_lab.dashboard import build_lab_state


class BuildLabStatePathResolutionTests(unittest.TestCase):
    def test_build_lab_state_resolves_dataset_paths_from_project_root(self):
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

            state = build_lab_state(root / "ai_lab" / "configs" / "config.yaml", root / "runs" / "manifest.json")

            self.assertEqual(state["dataset_counts"]["train"], 1)
            self.assertEqual(state["dataset_counts"]["eval"], 1)
            self.assertEqual(state["status"], "ready")


if __name__ == "__main__":
    unittest.main()
