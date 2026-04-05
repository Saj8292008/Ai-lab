from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_run_manifest(
    project_name: str,
    base_model: str,
    dataset_path: str,
    output_dir: str,
    created_at: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "project_name": project_name,
        "base_model": base_model,
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "created_at": created_at or _now_iso(),
        "tags": list(tags or []),
    }


def recommend_recipe(hardware: str, focus: str) -> dict[str, Any]:
    hardware_key = hardware.strip().lower()
    focus_key = focus.strip().lower()

    if focus_key != "text":
        return {
            "trainer": "lora",
            "precision": "fp16-or-bf16",
            "recommended_models": ["stabilityai/stable-diffusion-xl-base-1.0"],
            "notes": ["start with a narrow dataset and visual eval set"],
        }

    if hardware_key in {"apple-silicon-16gb", "apple-silicon", "m1-16gb", "m2-16gb"}:
        return {
            "trainer": "peft-lora",
            "precision": "float32-or-bf16-when-stable",
            "recommended_models": [
                "Qwen/Qwen3-1.7B",
                "Qwen/Qwen2.5-1.5B-Instruct",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            ],
            "notes": [
                "prefer short context windows first",
                "keep batch size tiny and use gradient accumulation",
                "treat the laptop as a research and validation box, not a full production trainer",
            ],
        }

    return {
        "trainer": "qlora",
        "precision": "4bit-base-16bit-adapters",
        "recommended_models": [
            "Qwen/Qwen2.5-7B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
        "notes": [
            "collect a fixed eval set before the first training run",
            "log hyperparameters and compare against the untuned base model",
        ],
    }
