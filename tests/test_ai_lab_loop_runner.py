import json
import tempfile
import unittest
from pathlib import Path

from ai_lab.loop_runner import run_phase_one_loops


class RunPhaseOneLoopsTests(unittest.TestCase):
    def test_run_phase_one_loops_creates_reports_for_all_default_loops(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "ai_lab" / "configs").mkdir(parents=True)
            config_path = root / "ai_lab" / "configs" / "config.yaml"
            config_path.write_text(
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
            manifest_path = root / "runs" / "manifest.json"
            manifest_path.write_text(json.dumps({"created_at": "2026-03-21T22:15:00Z"}))

            result = run_phase_one_loops(config_path, manifest_path, report_dir=root / "reports", created_at="2026-03-22T09:00:00Z")

            self.assertEqual(result["report_count"], 3)
            self.assertEqual(result["loop_ids"], ["prompt-optimizer", "rd-council", "ambient-research"])
            self.assertEqual(len(list((root / "reports").glob("*.json"))), 3)


if __name__ == "__main__":
    unittest.main()
