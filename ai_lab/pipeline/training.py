from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ai_lab.lab_manifest import build_run_manifest, recommend_recipe
from ai_lab.pipeline.cloud import build_cloud_training_bundle
from ai_lab.pipeline.synthetic import SyntheticExample, write_jsonl


def build_peft_training_artifacts(
    *,
    output_dir: str | Path,
    project_name: str,
    base_model: str,
    hardware_profile: str,
    synthetic_dataset_path: str | Path | None = None,
    train_epochs: int = 1,
    learning_rate: float = 2e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    target_modules: Iterable[str] | None = None,
    cloud_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    manifest = build_run_manifest(
        project_name=project_name,
        base_model=base_model,
        dataset_path=str(synthetic_dataset_path or output / "train.jsonl"),
        output_dir=str(output),
        tags=["sft", "lora", "synthetic"],
    )
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    training_config = {
        "project_name": project_name,
        "base_model": base_model,
        "hardware_profile": hardware_profile,
        "dataset_path": str(synthetic_dataset_path or output / "train.jsonl"),
        "output_dir": str(output),
        "train_epochs": train_epochs,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "target_modules": list(target_modules or ["q_proj", "v_proj"]),
    }
    training_config_path = output / "training_config.json"
    training_config_path.write_text(json.dumps(training_config, indent=2, sort_keys=True))

    result = {
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "training_config": training_config,
        "training_config_path": str(training_config_path),
        "dataset_path": str(synthetic_dataset_path or output / "train.jsonl"),
        "recipe": recommend_recipe(hardware_profile, "text"),
        "run_command": f"python3 -m ai_lab.scripts.train_peft_sft --config {training_config_path}",
    }

    if cloud_config and bool(cloud_config.get("enabled", False)):
        result["cloud"] = build_cloud_training_bundle(
            output_dir=output,
            training_config=training_config,
            cloud_config=cloud_config,
        )

    return result


def build_lora_training_artifacts(
    *,
    output_dir: str | Path,
    project_name: str,
    base_model: str,
    hardware_profile: str,
    synthetic_examples: list[SyntheticExample],
    cloud_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    train_path = write_jsonl(output / "train.jsonl", synthetic_examples)
    result = build_peft_training_artifacts(
        output_dir=output,
        project_name=project_name,
        base_model=base_model,
        hardware_profile=hardware_profile,
        synthetic_dataset_path=train_path,
        cloud_config=cloud_config,
    )
    result["train_path"] = str(train_path)
    return result
