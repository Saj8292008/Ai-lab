import unittest

from ai_lab.lab_manifest import build_run_manifest, recommend_recipe


class BuildRunManifestTests(unittest.TestCase):
    def test_build_run_manifest_includes_core_metadata(self):
        manifest = build_run_manifest(
            project_name="sydney-lab",
            base_model="Qwen/Qwen3-1.7B",
            dataset_path="data/train.jsonl",
            output_dir="runs/qwen3-1.7b-lora",
            created_at="2026-03-21T22:15:00Z",
            tags=["karpathy", "local", "lora"],
        )

        self.assertEqual(manifest["project_name"], "sydney-lab")
        self.assertEqual(manifest["base_model"], "Qwen/Qwen3-1.7B")
        self.assertEqual(manifest["dataset_path"], "data/train.jsonl")
        self.assertEqual(manifest["output_dir"], "runs/qwen3-1.7b-lora")
        self.assertEqual(manifest["created_at"], "2026-03-21T22:15:00Z")
        self.assertEqual(manifest["tags"], ["karpathy", "local", "lora"])

    def test_build_run_manifest_normalizes_empty_tags(self):
        manifest = build_run_manifest(
            project_name="sydney-lab",
            base_model="Qwen/Qwen3-1.7B",
            dataset_path="data/train.jsonl",
            output_dir="runs/qwen3-1.7b-lora",
            created_at="2026-03-21T22:15:00Z",
        )

        self.assertEqual(manifest["tags"], [])


class RecommendRecipeTests(unittest.TestCase):
    def test_recommend_recipe_for_apple_silicon_text_focus(self):
        recipe = recommend_recipe(hardware="apple-silicon-16gb", focus="text")

        self.assertEqual(recipe["trainer"], "peft-lora")
        self.assertEqual(recipe["precision"], "float32-or-bf16-when-stable")
        self.assertIn("Qwen/Qwen3-1.7B", recipe["recommended_models"])
        self.assertIn("prefer short context windows first", recipe["notes"])

    def test_recommend_recipe_for_nvidia_text_focus_prefers_qlora(self):
        recipe = recommend_recipe(hardware="nvidia-24gb", focus="text")

        self.assertEqual(recipe["trainer"], "qlora")
        self.assertEqual(recipe["precision"], "4bit-base-16bit-adapters")
        self.assertIn("Qwen/Qwen2.5-7B-Instruct", recipe["recommended_models"])


if __name__ == "__main__":
    unittest.main()
