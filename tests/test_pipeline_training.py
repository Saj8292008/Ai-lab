import tempfile
import unittest
from pathlib import Path

from ai_lab.pipeline.synthetic import SyntheticExample
from ai_lab.pipeline.training import build_lora_training_artifacts


class TrainingPrepTests(unittest.TestCase):
    def test_build_lora_training_artifacts_writes_manifest_and_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts = build_lora_training_artifacts(
                output_dir=root / "training",
                project_name="sydney-ai-research-lab",
                base_model="Qwen/Qwen3-1.7B",
                hardware_profile="apple-silicon-16gb",
                synthetic_examples=[
                    SyntheticExample(
                        instruction="Summarize the product roadmap.",
                        input_text="",
                        output_text="The first three products are a private business assistant LoRA, a niche creator style/model pack, and a local knowledge worker copilot.",
                    )
                ],
            )

            self.assertTrue((root / "training" / "manifest.json").exists())
            self.assertTrue((root / "training" / "train.jsonl").exists())
            self.assertIn("recipe", artifacts)


if __name__ == "__main__":
    unittest.main()

