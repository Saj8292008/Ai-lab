from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ai_lab.pipeline.retrieval import SearchHit


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    prompt: str
    reference_answer: str
    expected_terms: tuple[str, ...] = ()
    source_doc_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalResult:
    task_id: str
    variant: str
    answer: str
    score: float
    metrics: dict[str, float]
    retrieved_chunks: list[dict[str, Any]]
    created_at: str


def load_eval_tasks(path: str | Path) -> list[EvalTask]:
    file = Path(path)
    if not file.exists():
        return []

    tasks: list[EvalTask] = []
    for row in file.read_text().splitlines():
        if not row.strip():
            continue
        payload = json.loads(row)
        tasks.append(
            EvalTask(
                task_id=str(payload["task_id"]),
                prompt=str(payload["prompt"]),
                reference_answer=str(payload.get("reference_answer", "")),
                expected_terms=tuple(str(term) for term in payload.get("expected_terms", [])),
                source_doc_ids=tuple(str(doc_id) for doc_id in payload.get("source_doc_ids", [])),
                metadata={k: v for k, v in payload.items() if k not in {"task_id", "prompt", "reference_answer", "expected_terms", "source_doc_ids"}},
            )
        )
    return tasks


def score_answer(task: EvalTask, answer: str, retrieved_hits: Iterable[SearchHit] | None = None) -> dict[str, float]:
    answer_terms = set(_tokenize(answer))
    reference_terms = set(_tokenize(task.reference_answer))
    expected_terms = {term.lower() for term in task.expected_terms}

    answer_overlap = _coverage(expected_terms, answer_terms)
    reference_overlap = _coverage(reference_terms, answer_terms)
    support_overlap = 0.0

    hits = list(retrieved_hits or [])
    if hits and task.source_doc_ids:
        support_overlap = 1.0 if any(hit.chunk.doc_id in task.source_doc_ids for hit in hits) else 0.0

    score = round((0.45 * answer_overlap) + (0.35 * reference_overlap) + (0.20 * support_overlap), 4)
    return {
        "answer_overlap": round(answer_overlap, 4),
        "reference_overlap": round(reference_overlap, 4),
        "support_overlap": round(support_overlap, 4),
        "score": score,
    }


def _coverage(required_terms: set[str], actual_terms: set[str]) -> float:
    if not required_terms:
        return 0.0
    matches = sum(1 for term in required_terms if term in actual_terms)
    return matches / len(required_terms)


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())

