from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ai_lab.pipeline.evals import EvalResult, EvalTask


@dataclass(frozen=True)
class SyntheticExample:
    instruction: str
    input_text: str
    output_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def build_synthetic_examples(
    tasks: Iterable[EvalTask],
    results: Iterable[EvalResult],
    *,
    min_score: float = 0.72,
) -> list[SyntheticExample]:
    task_map = {task.task_id: task for task in tasks}
    results_by_task: dict[str, list[EvalResult]] = {}
    for result in results:
        results_by_task.setdefault(result.task_id, []).append(result)

    examples: list[SyntheticExample] = []

    for task_id, task_results in results_by_task.items():
        task_results = sorted(task_results, key=lambda item: item.score, reverse=True)
        if not task_results:
            continue
        best_result = task_results[0]
        baseline_result = next((item for item in task_results if item.variant == "baseline"), None)
        baseline_score = baseline_result.score if baseline_result else 0.0
        improvement = best_result.score - baseline_score
        if best_result.score < min_score and improvement < 0.05:
            continue

        task = task_map.get(task_id)
        if not task:
            continue
        examples.append(
            SyntheticExample(
                instruction=task.prompt,
                input_text="",
                output_text=best_result.answer,
                metadata={
                    "task_id": best_result.task_id,
                    "variant": best_result.variant,
                    "score": best_result.score,
                    "baseline_score": baseline_score,
                    "improvement": round(improvement, 4),
                },
            )
        )

    return examples


def write_jsonl(path: str | Path, examples: Iterable[SyntheticExample]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "instruction": example.instruction,
                "input": example.input_text,
                "output": example.output_text,
                **example.metadata,
            },
            sort_keys=True,
        )
        for example in examples
    ]
    output.write_text("\n".join(lines) + ("\n" if lines else ""))
    return output
