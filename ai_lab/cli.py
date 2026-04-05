from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai_lab.app import main as serve_main
from ai_lab.config import load_lab_config
from ai_lab.paths import repo_root
from ai_lab.pipeline.chunking import chunk_documents
from ai_lab.pipeline.experiments import run_experiment
from ai_lab.pipeline.ingest import load_documents
from ai_lab.pipeline.retrieval import SparseVectorIndex
from ai_lab.storage import LabStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Sydney AI Lab command line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the default eval-driven experiment loop.")
    run_parser.add_argument("--config", default="config/lab.sample.yaml")

    query_parser = subparsers.add_parser("query", help="Query the local retrieval index.")
    query_parser.add_argument("prompt")
    query_parser.add_argument("--config", default="config/lab.sample.yaml")
    query_parser.add_argument("--top-k", type=int, default=4)

    serve_parser = subparsers.add_parser("serve", help="Launch the dashboard server.")
    serve_parser.add_argument("--config", default="config/lab.sample.yaml")
    serve_parser.add_argument("--manifest", default="storage/runs/latest-manifest.json")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8787)

    args = parser.parse_args()

    if args.command == "serve":
        sys.argv = [
            "ai_lab.app",
            "--config",
            args.config,
            "--manifest",
            args.manifest,
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
        serve_main()
        return

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root() / config_path
    config = load_lab_config(config_path)

    if args.command == "run":
        summary = run_experiment(config=config, store=LabStore(config_path.parent.parent / "storage"))
        print(json.dumps(summary, indent=2))
        return

    if args.command == "query":
        docs_dir = Path(config.get("docs_dir", "storage/docs"))
        if not docs_dir.is_absolute():
            docs_dir = repo_root() / docs_dir
        docs = load_documents(docs_dir)
        chunks = chunk_documents(
            docs,
            chunk_words=int(config.get("chunk_words", 180)),
            overlap_words=int(config.get("chunk_overlap", 40)),
        )
        index = SparseVectorIndex.build(chunks)
        hits = index.search(args.prompt, top_k=args.top_k)
        payload = [
            {
                "score": round(hit.score, 4),
                "doc_id": hit.chunk.doc_id,
                "path": hit.chunk.path,
                "title": hit.chunk.title,
                "preview": hit.chunk.text[:220],
                "matched_terms": list(hit.matched_terms),
            }
            for hit in hits
        ]
        print(json.dumps(payload, indent=2))
        return


if __name__ == "__main__":
    main()

