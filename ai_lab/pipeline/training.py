from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_lab.lab_manifest import build_run_manifest, recommend_recipe
from ai_lab.pipeline.synthetic import SyntheticExample, write_jsonl


def build_lora_training_artifacts(
    *,
    output_dir: str | Path,
    project_name: str,
    base_model: str,
    hardware_profile: str,
    synthetic_examples: list[SyntheticExample],
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_path = write_jsonl(output / "train.jsonl", synthetic_examples)
    manifest = build_run_manifest(
        project_name=project_name,
        base_model=base_model,
        dataset_path=str(train_path),
        output_dir=str(output),
        tags=["sft", "lora", "synthetic"],
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "train_path": str(train_path),
        "recipe": recommend_recipe(hardware_profile, "text"),
    }
