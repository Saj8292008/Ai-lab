from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable

import re

from ai_lab.pipeline.ingest import Document


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    path: str
    title: str
    text: str
    ordinal: int
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_documents(
    documents: Iterable[Document],
    *,
    chunk_words: int = 180,
    overlap_words: int = 40,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(_chunk_document(document, chunk_words=chunk_words, overlap_words=overlap_words))
    return chunks


def _chunk_document(document: Document, *, chunk_words: int, overlap_words: int) -> list[Chunk]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", document.text.strip()) if part.strip()]
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_words = 0
    ordinal = 0

    def flush_buffer() -> None:
        nonlocal buffer, buffer_words, ordinal
        if not buffer:
            return
        text = "\n\n".join(buffer).strip()
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(document.doc_id, ordinal, text),
                doc_id=document.doc_id,
                path=document.path,
                title=document.title,
                text=text,
                ordinal=ordinal,
                metadata={
                    "source_type": document.source_type,
                    "paragraph_count": len(buffer),
                    **document.metadata,
                },
            )
        )
        ordinal += 1
        if overlap_words and buffer:
            tail = _tail_words(buffer, overlap_words)
            buffer = [tail] if tail else []
            buffer_words = len(_tokenize(tail)) if tail else 0
        else:
            buffer = []
            buffer_words = 0

    for paragraph in paragraphs:
        paragraph_words = len(_tokenize(paragraph))
        if paragraph_words > chunk_words:
            if buffer:
                flush_buffer()
            long_chunks = _split_long_paragraph(
                document,
                paragraph,
                ordinal_start=ordinal,
                chunk_words=chunk_words,
                overlap_words=overlap_words,
            )
            chunks.extend(long_chunks)
            if long_chunks:
                ordinal = long_chunks[-1].ordinal + 1
            continue

        if buffer_words and buffer_words + paragraph_words > chunk_words:
            flush_buffer()

        buffer.append(paragraph)
        buffer_words += paragraph_words

    flush_buffer()
    return chunks


def _split_long_paragraph(
    document: Document,
    paragraph: str,
    *,
    ordinal_start: int,
    chunk_words: int,
    overlap_words: int,
) -> list[Chunk]:
    words = _tokenize(paragraph)
    if not words:
        return []

    chunks: list[Chunk] = []
    start = 0
    ordinal = ordinal_start
    step = max(chunk_words - overlap_words, 1)

    while start < len(words):
        window = words[start : start + chunk_words]
        text = " ".join(window)
        chunks.append(
            Chunk(
                chunk_id=_chunk_id(document.doc_id, ordinal, text),
                doc_id=document.doc_id,
                path=document.path,
                title=document.title,
                text=text,
                ordinal=ordinal,
                metadata={
                    "source_type": document.source_type,
                    "paragraph_count": 1,
                    **document.metadata,
                },
            )
        )
        ordinal += 1
        start += step
    return chunks


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _tail_words(paragraphs: list[str], overlap_words: int) -> str:
    words = _tokenize("\n\n".join(paragraphs))
    return " ".join(words[-overlap_words:]) if overlap_words and words else ""


def _chunk_id(doc_id: str, ordinal: int, text: str) -> str:
    return sha1(f"{doc_id}:{ordinal}:{text}".encode("utf-8")).hexdigest()[:16]
