from __future__ import annotations

import json
from time import sleep
from typing import Any, Iterator

from ai_lab.paths import storage_root
from ai_lab.pipeline.cache import ensure_cached_index, fingerprint_documents
from ai_lab.pipeline.experiments import run_experiment
from ai_lab.pipeline.ingest import load_documents
from ai_lab.storage import LabStore, now_iso


def refresh_document_cache(config: dict[str, Any], store: LabStore | None = None) -> dict[str, Any]:
    store = store or LabStore(storage_root())
    lab_root = store.root

    docs_dir = config.get("docs_dir", lab_root / "docs")
    docs = load_documents(docs_dir)
    if not docs:
        raise FileNotFoundError(f"No ingestible docs found in {docs_dir}")

    chunk_words = int(config.get("chunk_words", 180))
    overlap_words = int(config.get("chunk_overlap", 40))
    fingerprint = fingerprint_documents(
        docs,
        chunk_words=chunk_words,
        overlap_words=overlap_words,
        salt=json.dumps(config.get("index_salt", ""), sort_keys=True),
    )
    cached = ensure_cached_index(
        lab_root / "indexes",
        fingerprint,
        documents=docs,
        chunk_words=chunk_words,
        overlap_words=overlap_words,
        metadata={
            "chunk_words": chunk_words,
            "overlap_words": overlap_words,
            "docs_dir": str(docs_dir),
            "updated_at": now_iso(),
        },
    )

    snapshot = {
        "created_at": now_iso(),
        "docs_dir": str(docs_dir),
        "docs": len(docs),
        "chunks": len(cached.chunks),
        "fingerprint": fingerprint,
        "cache_hit": cached.hit,
        "cache_dir": str(cached.cache_dir),
    }
    store.write_json_artifact("watch", "latest.json", snapshot)
    return snapshot


def iter_document_watch(
    config: dict[str, Any],
    *,
    store: LabStore | None = None,
    interval_seconds: float = 15.0,
    run_on_change: bool = False,
    once: bool = False,
) -> Iterator[dict[str, Any]]:
    store = store or LabStore(storage_root())
    previous_fingerprint: str | None = None

    while True:
        snapshot = refresh_document_cache(config, store=store)
        snapshot["changed"] = snapshot["fingerprint"] != previous_fingerprint
        snapshot["run_on_change"] = run_on_change

        if snapshot["changed"] and run_on_change:
            summary = run_experiment(config=config, store=store)
            snapshot["run_id"] = summary["run_id"]
            snapshot["run_created_at"] = summary["created_at"]
            snapshot["run_cache_hit"] = summary["cache_hit"]

        store.write_json_artifact("watch", "latest-event.json", snapshot)
        yield snapshot

        if once:
            return

        previous_fingerprint = snapshot["fingerprint"]
        sleep(interval_seconds)
