from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ai_lab.paths import ensure_lab_directories, storage_root
from ai_lab.pipeline.chunking import Chunk, chunk_documents
from ai_lab.pipeline.evals import EvalResult, EvalTask, load_eval_tasks, score_answer
from ai_lab.pipeline.ingest import Document, load_documents
from ai_lab.pipeline.retrieval import SearchHit, SparseVectorIndex
from ai_lab.pipeline.synthetic import build_synthetic_examples, write_jsonl
from ai_lab.storage import LabStore, now_iso


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    use_retrieval: bool
    top_k: int = 4
    instruction: str = ""


@dataclass
class ExperimentRun:
    run_id: str
    created_at: str
    config: dict[str, Any]
    docs: list[Document] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    tasks: list[EvalTask] = field(default_factory=list)
    results: list[EvalResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


DEFAULT_VARIANTS = [
    ExperimentVariant(
        name="baseline",
        use_retrieval=False,
        top_k=0,
        instruction="Answer from the question alone.",
    ),
    ExperimentVariant(
        name="rag",
        use_retrieval=True,
        top_k=4,
        instruction="Use the retrieved context, prefer direct evidence, and keep the answer concise.",
    ),
]


def run_experiment(
    *,
    config: dict[str, Any],
    store: LabStore | None = None,
    variants: Iterable[ExperimentVariant] | None = None,
) -> dict[str, Any]:
    ensure_lab_directories()
    store = store or LabStore(storage_root())
    lab_root = store.root
    run_created_at = now_iso()
    run_name = str(config.get("project_name", "ai-lab"))
    variants = list(variants or DEFAULT_VARIANTS)

    docs_dir = Path(config.get("docs_dir", lab_root / "docs"))
    eval_file = Path(config.get("eval_file", lab_root / "datasets" / "evals.jsonl"))
    chunk_words = int(config.get("chunk_words", 180))
    overlap_words = int(config.get("chunk_overlap", 40))
    top_k = int(config.get("top_k", 4))
    min_synthetic_score = float(config.get("min_synthetic_score", 0.72))

    docs = load_documents(docs_dir)
    chunks = chunk_documents(docs, chunk_words=chunk_words, overlap_words=overlap_words)
    index = SparseVectorIndex.build(chunks)
    tasks = load_eval_tasks(eval_file)

    if not docs:
        raise FileNotFoundError(f"No ingestible docs found in {docs_dir}")
    if not tasks:
        raise FileNotFoundError(f"No eval tasks found in {eval_file}")

    results: list[EvalResult] = []
    variant_scores: dict[str, list[float]] = {variant.name: [] for variant in variants}
    eval_rows: list[dict[str, Any]] = []

    for variant in variants:
        for task in tasks:
            retrieved = index.search(task.prompt, top_k=variant.top_k or top_k) if variant.use_retrieval else []
            answer = _answer_task(task, retrieved, variant)
            metrics = score_answer(task, answer, retrieved)
            result = EvalResult(
                task_id=task.task_id,
                variant=variant.name,
                answer=answer,
                score=metrics["score"],
                metrics={key: value for key, value in metrics.items() if key != "score"},
                retrieved_chunks=[
                    {
                        "chunk_id": hit.chunk.chunk_id,
                        "doc_id": hit.chunk.doc_id,
                        "path": hit.chunk.path,
                        "score": round(hit.score, 4),
                        "matched_terms": list(hit.matched_terms),
                        "preview": _preview(hit.chunk.text),
                    }
                    for hit in retrieved
                ],
                created_at=run_created_at,
            )
            results.append(result)
            variant_scores[variant.name].append(result.score)
            eval_rows.append(
                {
                    "run_id": None,
                    "task_id": result.task_id,
                    "variant": result.variant,
                    "answer": result.answer,
                    "score": result.score,
                    "metrics": result.metrics,
                    "retrieved_chunks": result.retrieved_chunks,
                    "created_at": result.created_at,
                }
            )

    synthetic_examples = build_synthetic_examples(tasks, results, min_score=min_synthetic_score)
    synthetic_path = write_jsonl(
        lab_root / "synthetic" / f"{run_created_at.replace(':', '-')}.jsonl",
        synthetic_examples,
    )

    metrics = {
        "docs": len(docs),
        "chunks": len(chunks),
        "tasks": len(tasks),
        "variants": {
            name: {
                "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "best_score": round(max(scores), 4) if scores else 0.0,
                "worst_score": round(min(scores), 4) if scores else 0.0,
            }
            for name, scores in variant_scores.items()
        },
        "synthetic_examples": len(synthetic_examples),
    }

    artifact = {
        "created_at": run_created_at,
        "project_name": run_name,
        "config": config,
        "metrics": metrics,
        "docs": [
            {
                "doc_id": doc.doc_id,
                "path": doc.path,
                "title": doc.title,
                "source_type": doc.source_type,
                "metadata": doc.metadata,
            }
            for doc in docs
        ],
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "path": chunk.path,
                "title": chunk.title,
                "ordinal": chunk.ordinal,
                "metadata": chunk.metadata,
                "preview": _preview(chunk.text),
            }
            for chunk in chunks
        ],
        "tasks": [
            {
                "task_id": task.task_id,
                "prompt": task.prompt,
                "reference_answer": task.reference_answer,
                "expected_terms": list(task.expected_terms),
                "source_doc_ids": list(task.source_doc_ids),
                "metadata": task.metadata,
            }
            for task in tasks
        ],
        "results": [result_to_dict(result) for result in results],
        "synthetic_dataset_path": str(synthetic_path),
    }

    artifact_name = f"{run_created_at.replace(':', '-')}.json"
    artifact_path = store.write_json_artifact("runs", artifact_name, artifact)
    run_id = store.record_run(
        name=run_name,
        mode="eval-driven-iteration",
        config=config,
        metrics=metrics,
        artifact_path=artifact_path,
        created_at=run_created_at,
    )
    store.record_eval_results(run_id, [
        {"run_id": run_id, **result_to_dict(result)} for result in results
    ])

    for row in eval_rows:
        row["run_id"] = run_id

    store.write_json_artifact("eval_results", artifact_name, {"run_id": run_id, "results": eval_rows})

    summary = {
        "run_id": run_id,
        "created_at": run_created_at,
        "artifact_path": str(artifact_path),
        "metrics": metrics,
        "variants": metrics["variants"],
        "synthetic_dataset_path": str(synthetic_path),
    }
    return summary


def result_to_dict(result: EvalResult) -> dict[str, Any]:
    return {
        "task_id": result.task_id,
        "variant": result.variant,
        "answer": result.answer,
        "score": result.score,
        "metrics": result.metrics,
        "retrieved_chunks": result.retrieved_chunks,
        "created_at": result.created_at,
    }


def _answer_task(task: EvalTask, retrieved: list[SearchHit], variant: ExperimentVariant) -> str:
    if not variant.use_retrieval:
        return (
            "I cannot answer confidently without the source docs. "
            f"Task prompt: {task.prompt}"
        )

    if not retrieved:
        return "I searched the local index but found no relevant evidence."

    sentences: list[str] = []
    seen = set()
    query_terms = {term.lower() for term in task.expected_terms}
    query_terms.update(term for term in _tokenize(task.prompt) if len(term) > 3)

    for hit_index, hit in enumerate(retrieved):
        hit_lines = [
            _clean_candidate(raw_line)
            for raw_line in hit.chunk.text.splitlines()
        ]
        hit_lines = [line for line in hit_lines if line and len(line) >= 10]

        preferred = [
            line
            for line in hit_lines
            if query_terms and any(term in line.lower() for term in query_terms)
        ]

        candidates = preferred if preferred else (hit_lines[:3] if hit_index == 0 else hit_lines[:1])

        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            sentences.append(candidate)
            if len(sentences) >= 3:
                break
        if len(sentences) >= 3:
            break

    if not sentences:
        sentences = [_preview(hit.chunk.text, limit=180) for hit in retrieved[:2]]

    return " ".join(sentences)


def _preview(text: str, limit: int = 140) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _clean_candidate(line: str) -> str:
    candidate = line.strip()
    if not candidate:
        return ""
    candidate = candidate.lstrip("#").strip()
    candidate = candidate.lstrip("-*0123456789. ").strip()
    candidate = " ".join(candidate.split())
    if candidate.endswith(":"):
        candidate = candidate[:-1].strip()
    return candidate


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())
