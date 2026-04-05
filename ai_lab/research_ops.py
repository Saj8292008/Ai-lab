from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_default_loops(project_name: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "prompt-optimizer",
            "name": f"{project_name} prompt optimization",
            "cadence": "every 5m",
            "status": "active",
            "goal": "Continuously improve operating prompts and compare variants.",
        },
        {
            "id": "rd-council",
            "name": f"{project_name} R&D council",
            "cadence": "twice daily",
            "status": "active",
            "goal": "Have multiple specialist agents review progress and propose next experiments.",
        },
        {
            "id": "ambient-research",
            "name": f"{project_name} ambient research",
            "cadence": "every 30m",
            "status": "active",
            "goal": "Track opportunities, model updates, and startup-relevant research in the background.",
        },
    ]


def build_report(
    loop_id: str,
    title: str,
    summary: str,
    findings: list[str],
    next_actions: list[str],
    created_at: str | None = None,
) -> dict[str, Any]:
    return {
        "loop_id": loop_id,
        "title": title,
        "summary": summary,
        "findings": list(findings),
        "next_actions": list(next_actions),
        "created_at": created_at or _now_iso(),
    }


def write_report(report_dir: str | Path, report: dict[str, Any]) -> Path:
    target_dir = Path(report_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    created_at = str(report["created_at"]).replace(":", "-")
    path = target_dir / f"{created_at}--{report['loop_id']}.json"
    path.write_text(json.dumps(report, indent=2))
    return path


def load_reports(report_dir: str | Path, limit: int = 5) -> list[dict[str, Any]]:
    target_dir = Path(report_dir)
    if not target_dir.exists():
        return []

    reports = []
    for path in sorted(target_dir.glob("*.json"), reverse=True):
        reports.append(json.loads(path.read_text()))
        if len(reports) >= limit:
            break
    return reports
