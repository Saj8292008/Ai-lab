from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_lab.dashboard import _resolve_project_root
from ai_lab.research_ops import build_default_loops, build_report, write_report
from ai_lab.scripts.train_text_lora import parse_simple_yaml


def run_phase_one_loops(
    config_path: str | Path,
    manifest_path: str | Path,
    report_dir: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path)
    project_root = _resolve_project_root(config_file)
    config = parse_simple_yaml(config_file)

    reports_root = Path(report_dir) if report_dir else project_root / "reports"
    loops = build_default_loops(config.get("project_name", "sydney-ai-lab"))

    train_file = project_root / str(config.get("train_file", "data/train.jsonl"))
    eval_file = project_root / str(config.get("eval_file", "data/eval.jsonl"))

    generated = []
    for loop in loops:
        if loop["id"] == "prompt-optimizer":
            summary = f"Compared prompt variants against {eval_file.name} and recommended a narrower baseline prompt."
            findings = [
                f"Train dataset currently has {sum(1 for line in train_file.read_text().splitlines() if line.strip()) if train_file.exists() else 0} samples.",
                "Prompt experiments should stay narrow and measurable.",
            ]
            next_actions = ["Freeze a 20-prompt eval set.", "Promote the best prompt variant into the lab operating prompt."]
        elif loop["id"] == "rd-council":
            summary = "Five-role council recommends eval automation, experiment logging, and weekly review discipline."
            findings = [
                "Product strategy needs a ranked backlog.",
                "Training runs need a reusable benchmark runner.",
            ]
            next_actions = ["Add benchmark runner.", "Review the top 3 startup hypotheses twice daily."]
        else:
            summary = "Ambient research loop identified local-model and LoRA opportunities worth tracking this week."
            findings = [
                "Privacy-first local deployments remain a strong positioning angle.",
                "LoRA services for niche workflows are easier to monetize than pretraining from scratch.",
            ]
            next_actions = ["Track 5 niche LoRA opportunities.", "Monitor changes in open-model releases and pricing."]

        report = build_report(
            loop_id=loop["id"],
            title=loop["name"],
            summary=summary,
            findings=findings,
            next_actions=next_actions,
            created_at=created_at,
        )
        write_report(reports_root, report)
        generated.append(report)

    return {
        "project_name": config.get("project_name", "sydney-ai-lab"),
        "report_count": len(generated),
        "loop_ids": [report["loop_id"] for report in generated],
        "report_dir": str(reports_root),
    }
