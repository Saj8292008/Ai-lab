import tempfile
import unittest
from pathlib import Path

from ai_lab.pipeline.synthetic import SyntheticExample
from ai_lab.pipeline.training import build_lora_training_artifacts


class CloudBundleTests(unittest.TestCase):
    def test_build_lora_training_artifacts_emits_cloud_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifacts = build_lora_training_artifacts(
                output_dir=root / "training",
                project_name="sydney-ai-research-lab",
                base_model="Qwen/Qwen3-1.7B",
                hardware_profile="apple-silicon-16gb",
                synthetic_examples=[
                    SyntheticExample(
                        instruction="Summarize the roadmap.",
                        input_text="",
                        output_text="The first product is a private business assistant LoRA.",
                    )
                ],
                cloud_config={
                    "enabled": True,
                    "provider": "ssh",
                    "host": "training-host",
                    "user": "ubuntu",
                    "remote_root": "/workspace/ai-lab",
                    "bundle_name": "qwen3-run",
                },
            )

            self.assertIn("cloud", artifacts)
            self.assertTrue((root / "training" / "cloud" / "cloud_manifest.json").exists())
            self.assertTrue((root / "training" / "cloud" / "launch.sh").exists())
            self.assertTrue((root / "training" / "cloud" / "sync.sh").exists())
            self.assertIn("/workspace/ai-lab/qwen3-run", artifacts["cloud"]["manifest"]["remote_bundle_dir"])


if __name__ == "__main__":
    unittest.main()
