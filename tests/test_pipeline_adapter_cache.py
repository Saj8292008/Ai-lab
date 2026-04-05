import tempfile
import unittest
from pathlib import Path

from ai_lab.model_adapters import resolve_model_adapter
from ai_lab.pipeline.cache import fingerprint_documents, load_cached_index, save_cached_index
from ai_lab.pipeline.chunking import chunk_documents
from ai_lab.pipeline.ingest import load_documents
from ai_lab.pipeline.retrieval import SparseVectorIndex


class AdapterTests(unittest.TestCase):
    def test_heuristic_adapter_uses_retrieved_context(self) -> None:
        adapter = resolve_model_adapter({"model": {"provider": "heuristic", "name": "demo"}})
        prompt = (
            "Task: What are the first three lab products to test?\n"
            "Retrieved context:\n"
            "===\n"
            "The first three lab products to test are a private business assistant LoRA.\n"
            "---\n"
            "===\n"
            "Return 1-3 concise sentences with the answer only."
        )

        result = adapter.generate(prompt)

        self.assertEqual(result.provider, "heuristic")
        self.assertIn("private business assistant LoRA", result.text)


class CacheTests(unittest.TestCase):
    def test_cached_index_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "one.md").write_text("# One\n\nAlpha beta gamma.\n")
            (docs_dir / "two.md").write_text("# Two\n\nDelta epsilon zeta.\n")

            docs = load_documents(docs_dir)
            fingerprint = fingerprint_documents(docs, chunk_words=40, overlap_words=5)
            chunks = chunk_documents(docs, chunk_words=40, overlap_words=5)
            index = SparseVectorIndex.build(chunks)
            saved = save_cached_index(root / "indexes", fingerprint, documents=docs, chunks=chunks, index=index)

            cached = load_cached_index(root / "indexes", fingerprint)

            self.assertFalse(saved.hit)
            self.assertIsNotNone(cached)
            self.assertTrue(cached.hit)
            self.assertEqual(len(cached.chunks), len(chunks))
            self.assertGreater(len(cached.index.search("alpha")), 0)


if __name__ == "__main__":
    unittest.main()

