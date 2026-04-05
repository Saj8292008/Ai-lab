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
    rubric: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalResult:
    task_id: str
    variant: str
    answer: str
    score: float
    passed: bool
    metrics: dict[str, Any]
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
                rubric=dict(payload.get("rubric", {})),
                metadata={
                    k: v
                    for k, v in payload.items()
                    if k
                    not in {
                        "task_id",
                        "prompt",
                        "reference_answer",
                        "expected_terms",
                        "source_doc_ids",
                        "rubric",
                    }
                },
            )
        )
    return tasks


def score_answer(task: EvalTask, answer: str, retrieved_hits: Iterable[SearchHit] | None = None) -> dict[str, Any]:
    answer_terms = set(_tokenize(answer))
    reference_terms = set(_tokenize(task.reference_answer))
    rubric = task.rubric or {}
    required_terms = {str(term).lower() for term in rubric.get("required_terms", task.expected_terms)}
    forbidden_terms = {str(term).lower() for term in rubric.get("forbidden_terms", [])}
    required_doc_ids = {str(doc_id) for doc_id in rubric.get("required_doc_ids", task.source_doc_ids)}
    must_use_retrieval = bool(rubric.get("must_use_retrieval", bool(required_doc_ids)))
    required_coverage = float(rubric.get("required_coverage", 0.85))
    pass_threshold = float(rubric.get("pass_threshold", 0.7))
    min_retrieved_docs = int(rubric.get("min_retrieved_docs", 1 if must_use_retrieval else 0))

    answer_text = _normalize_text(answer)
    term_coverage = _phrase_coverage(required_terms, answer_text) if required_terms else 1.0
    reference_overlap = _coverage(reference_terms, answer_terms) if reference_terms else 0.0
    forbidden_hits = [term for term in forbidden_terms if _normalize_text(term) in answer_text]

    hits = list(retrieved_hits or [])
    support_doc_ids = {hit.chunk.doc_id for hit in hits}
    doc_support = 1.0 if required_doc_ids and required_doc_ids.intersection(support_doc_ids) else 0.0
    retrieval_support = 1.0 if len(hits) >= min_retrieved_docs else 0.0
    if not must_use_retrieval:
        retrieval_support = 1.0

    score = (
        (0.45 * term_coverage)
        + (0.25 * reference_overlap)
        + (0.20 * doc_support)
        + (0.10 * retrieval_support)
        - (0.30 * (1.0 if forbidden_hits else 0.0))
    )
    score = max(0.0, min(1.0, score))
    passed = (
        term_coverage >= required_coverage
        and not forbidden_hits
        and retrieval_support >= 1.0
        and score >= pass_threshold
    )

    return {
        "term_coverage": round(term_coverage, 4),
        "reference_overlap": round(reference_overlap, 4),
        "doc_support": round(doc_support, 4),
        "retrieval_support": round(retrieval_support, 4),
        "forbidden_penalty": 1.0 if forbidden_hits else 0.0,
        "passed": passed,
        "score": round(score, 4),
    }


def _coverage(required_terms: set[str], actual_terms: set[str]) -> float:
    if not required_terms:
        return 0.0
    matches = sum(1 for term in required_terms if term in actual_terms)
    return matches / len(required_terms)


def _phrase_coverage(required_terms: set[str], actual_text: str) -> float:
    if not required_terms:
        return 0.0
    matches = sum(1 for term in required_terms if _normalize_text(term) in actual_text)
    return matches / len(required_terms)


def _normalize_text(text: str) -> str:
    return " ".join(_tokenize(text))


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())
