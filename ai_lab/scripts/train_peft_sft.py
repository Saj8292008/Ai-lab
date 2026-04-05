from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_lab.config import load_lab_config
from ai_lab.pipeline.training import build_peft_training_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local PEFT/LoRA SFT job from synthetic examples.")
    parser.add_argument("--config", default=None, help="JSON training config written by the lab, or a lab YAML config.")
    parser.add_argument("--synthetic", default=None, help="Path to a synthetic JSONL dataset with instruction/input/output rows.")
    parser.add_argument("--output-dir", default=None, help="Directory for artifacts and model outputs.")
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--hardware-profile", default=None)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--target-modules", default="q_proj,v_proj")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--cloud", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = _load_training_config(args)
    cloud_config = dict(config.get("cloud", {}) or {})
    cloud_config["enabled"] = bool(args.cloud or cloud_config.get("enabled", False))
    artifact_plan = build_peft_training_artifacts(
        output_dir=config["output_dir"],
        project_name=config["project_name"],
        base_model=config["base_model"],
        hardware_profile=config["hardware_profile"],
        synthetic_dataset_path=config["dataset_path"],
        train_epochs=config["train_epochs"],
        learning_rate=config["learning_rate"],
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        cloud_config=cloud_config,
    )

    plan = {
        "config": config,
        "artifacts": artifact_plan,
        "notes": [
            "This script will use a real PEFT stack when transformers, datasets, peft, and trl are installed.",
            "If you only want to verify the wiring, keep --dry-run on.",
        ],
    }
    print(json.dumps(plan, indent=2))

    if args.dry_run:
        return

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install: transformers datasets peft trl torch"
        ) from exc

    dataset = load_dataset("json", data_files=str(config["dataset_path"]), split="train")
    dataset = dataset.map(_format_example)

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(config["base_model"], **model_kwargs)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=0.05,
        target_modules=list(config["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=float(config["train_epochs"]),
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        save_strategy="epoch",
        logging_steps=1,
        report_to=[],
        bf16=use_cuda,
        fp16=not use_cuda,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,
    )
    trainer.train()
    trainer.model.save_pretrained(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


def _load_training_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.config:
        path = Path(args.config)
        if path.exists():
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text())
                config = dict(payload if isinstance(payload, dict) else {})
            else:
                config = load_lab_config(path)
        else:
            raise SystemExit(f"Training config not found: {path}")
    else:
        config = {}

    if args.synthetic:
        config["dataset_path"] = args.synthetic
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.project_name:
        config["project_name"] = args.project_name
    if args.base_model:
        config["base_model"] = args.base_model
    if args.hardware_profile:
        config["hardware_profile"] = args.hardware_profile

    config.setdefault("project_name", "sydney-ai-research-lab")
    config.setdefault("base_model", "Qwen/Qwen3-1.7B")
    config.setdefault("hardware_profile", "apple-silicon-16gb")
    config.setdefault("dataset_path", "storage/synthetic/latest.jsonl")
    config.setdefault("output_dir", "storage/artifacts/peft-run")
    config.setdefault("train_epochs", args.train_epochs)
    config.setdefault("learning_rate", args.learning_rate)
    config.setdefault("lora_r", args.lora_r)
    config.setdefault("lora_alpha", args.lora_alpha)
    config.setdefault("target_modules", [module.strip() for module in args.target_modules.split(",") if module.strip()])

    return config


def _format_example(example: dict[str, Any]) -> dict[str, str]:
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    output_text = str(example.get("output", "")).strip()
    text = "\n\n".join(
        part
        for part in [
            f"### Instruction:\n{instruction}" if instruction else "",
            f"### Input:\n{input_text}" if input_text else "",
            f"### Response:\n{output_text}" if output_text else "",
        ]
        if part
    ).strip()
    return {"text": text}


if __name__ == "__main__":
    main()
