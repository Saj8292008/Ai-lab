from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable

from ai_lab.pipeline.chunking import Chunk
from ai_lab.pipeline.ingest import Document
from ai_lab.pipeline.retrieval import SparseVectorIndex


@dataclass(frozen=True)
class CachedIndexBundle:
    fingerprint: str
    cache_dir: Path
    chunks: list[Chunk]
    index: SparseVectorIndex
    hit: bool


def fingerprint_documents(
    documents: Iterable[Document],
    *,
    chunk_words: int,
    overlap_words: int,
    salt: str = "",
) -> str:
    payload = {
        "chunk_words": chunk_words,
        "overlap_words": overlap_words,
        "salt": salt,
        "documents": [
            {
                "doc_id": doc.doc_id,
                "path": doc.path,
                "title": doc.title,
                "source_type": doc.source_type,
                "text": doc.text,
                "metadata": doc.metadata,
            }
            for doc in sorted(documents, key=lambda item: item.path)
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha1(raw).hexdigest()[:16]


def load_cached_index(cache_root: str | Path, fingerprint: str) -> CachedIndexBundle | None:
    cache_dir = Path(cache_root) / fingerprint
    manifest_path = cache_dir / "manifest.json"
    chunks_path = cache_dir / "chunks.json"
    index_path = cache_dir / "index.json"

    if not manifest_path.exists() or not chunks_path.exists() or not index_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("fingerprint") != fingerprint:
        return None

    chunks = [
        Chunk(
            chunk_id=str(item["chunk_id"]),
            doc_id=str(item["doc_id"]),
            path=str(item["path"]),
            title=str(item["title"]),
            text=str(item["text"]),
            ordinal=int(item["ordinal"]),
            metadata=dict(item.get("metadata", {})),
        )
        for item in json.loads(chunks_path.read_text())
    ]
    index = SparseVectorIndex.from_payload(json.loads(index_path.read_text()), chunks)
    return CachedIndexBundle(fingerprint=fingerprint, cache_dir=cache_dir, chunks=chunks, index=index, hit=True)


def save_cached_index(
    cache_root: str | Path,
    fingerprint: str,
    *,
    documents: Iterable[Document],
    chunks: Iterable[Chunk],
    index: SparseVectorIndex,
    metadata: dict[str, Any] | None = None,
) -> CachedIndexBundle:
    cache_dir = Path(cache_root) / fingerprint
    cache_dir.mkdir(parents=True, exist_ok=True)

    chunk_list = list(chunks)
    document_list = list(documents)
    manifest = {
        "fingerprint": fingerprint,
        "document_count": len(document_list),
        "chunk_count": len(chunk_list),
        "metadata": metadata or {},
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (cache_dir / "chunks.json").write_text(json.dumps([asdict(chunk) for chunk in chunk_list], indent=2, sort_keys=True))
    (cache_dir / "index.json").write_text(json.dumps(index.to_payload(), indent=2, sort_keys=True))
    return CachedIndexBundle(fingerprint=fingerprint, cache_dir=cache_dir, chunks=chunk_list, index=index, hit=False)

