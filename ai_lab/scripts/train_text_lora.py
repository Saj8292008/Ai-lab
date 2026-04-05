from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_lab.lab_manifest import build_run_manifest, recommend_recipe


def parse_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(0, data)]

    for raw_line in path.read_text().splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        while len(stack) > 1 and indent < stack[-1][0]:
            stack.pop()

        current = stack[-1][1]

        if line.startswith("- "):
            if not isinstance(current, list):
                raise ValueError(f"List item found outside list context: {line}")
            current.append(line[2:].strip())
            continue

        key, value = [part.strip() for part in line.split(":", 1)]
        if value == "":
            next_container: Any = [] if key == "notes" or key == "target_modules" else {}
            current[key] = next_container
            stack.append((indent + 2, next_container))
            continue

        if value.lower() in {"true", "false"}:
            parsed: Any = value.lower() == "true"
        else:
            try:
                parsed = int(value) if value.isdigit() else float(value)
            except ValueError:
                parsed = value
        current[key] = parsed

    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a local text LoRA run for Sydney AI Lab.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = parse_simple_yaml(config_path)
    recipe = recommend_recipe(config["hardware_profile"], config.get("focus", "text"))
    manifest = build_run_manifest(
        project_name=config["project_name"],
        base_model=config["base_model"],
        dataset_path=config["train_file"],
        output_dir=config["output_dir"],
        tags=["karpathy", "local", "lora", config.get("focus", "text")],
    )

    result = {
        "config": config,
        "recipe": recipe,
        "manifest": manifest,
        "next_steps": [
            "verify train and eval JSONL files exist",
            "hand-check 20 random samples before training",
            "run a base model eval before first finetune",
            "only then launch the LoRA training job",
        ],
    }

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    result["manifest_path"] = str(manifest_path)
    print(json.dumps(result, indent=2))

    if args.dry_run:
        return

    print("\nTraining launch is intentionally not automated yet.")
    print("Manifest written. Use this as a checkpoint before wiring in transformers/peft training.")


if __name__ == "__main__":
    main()
