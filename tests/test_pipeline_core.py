import tempfile
import unittest
from pathlib import Path

from ai_lab.pipeline.chunking import chunk_documents
from ai_lab.pipeline.ingest import load_documents
from ai_lab.pipeline.retrieval import SparseVectorIndex


class PipelineCoreTests(unittest.TestCase):
    def test_ingest_chunk_and_retrieve(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "notes.md").write_text(
                "# Research Notes\n\nThe first product is a private business assistant LoRA.\n\nThe cadence is daily inspection and weekly comparison.\n"
            )
            (docs_dir / "transcript.txt").write_text(
                "Apple Silicon is for tiny LoRA experiments and evals.\n"
            )

            documents = load_documents(docs_dir)
            chunks = chunk_documents(documents, chunk_words=40, overlap_words=5)
            index = SparseVectorIndex.build(chunks)
            hits = index.search("What is the private business assistant LoRA?", top_k=2)

            self.assertEqual(len(documents), 2)
            self.assertGreaterEqual(len(chunks), 2)
            self.assertGreaterEqual(len(hits), 1)
            self.assertIn("private business assistant", hits[0].chunk.text.lower())


if __name__ == "__main__":
    unittest.main()

