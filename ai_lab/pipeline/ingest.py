from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable


TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".rst", ".json", ".jsonl", ".srt", ".vtt"}


@dataclass(frozen=True)
class Document:
    doc_id: str
    path: str
    title: str
    text: str
    source_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_documents(source: str | Path, suffixes: Iterable[str] | None = None) -> list[Document]:
    root = Path(source)
    if not root.exists():
        return []

    allowed = {suffix.lower() for suffix in (suffixes or TEXT_SUFFIXES)}
    documents: list[Document] = []

    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.suffix.lower() not in allowed:
            continue

        text, source_type, metadata = _read_source(path)
        if not text.strip():
            continue

        rel_path = str(path.relative_to(root))
        documents.append(
            Document(
                doc_id=_stable_id(rel_path, text),
                path=rel_path,
                title=_extract_title(text, path),
                text=text.strip(),
                source_type=source_type,
                metadata={
                    "relative_path": rel_path,
                    "suffix": path.suffix.lower(),
                    "size_bytes": path.stat().st_size,
                    "source_name": path.name,
                    **metadata,
                },
            )
        )

    return documents


def _read_source(path: Path) -> tuple[str, str, dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        return json.dumps(payload, indent=2, sort_keys=True), "json", {}
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return json.dumps(rows, indent=2, sort_keys=True), "jsonl", {"row_count": len(rows)}
    return path.read_text(), "text", {}


def _extract_title(text: str, path: Path) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or path.stem
        return stripped[:80]
    return path.stem


def _stable_id(path: str, text: str) -> str:
    return sha1(f"{path}\n{text}".encode("utf-8")).hexdigest()[:16]

