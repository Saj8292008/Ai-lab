import unittest

from ai_lab.dashboard import build_lab_state


class LiveRepoStateTests(unittest.TestCase):
    def test_live_repo_reads_seed_datasets_as_ready(self):
        state = build_lab_state(
            "ai_lab/configs/qwen3_text_lora.yaml",
            "runs/qwen3-1.7b-lora/manifest.json",
        )

        self.assertGreaterEqual(state["dataset_counts"]["train"], 1)
        self.assertGreaterEqual(state["dataset_counts"]["eval"], 1)
        self.assertEqual(state["status"], "ready")


if __name__ == "__main__":
    unittest.main()
