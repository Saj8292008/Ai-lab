import tempfile
import unittest
from pathlib import Path

from ai_lab.pipeline.watching import iter_document_watch
from ai_lab.storage import LabStore


class WatcherTests(unittest.TestCase):
    def test_watch_refreshes_cache_and_writes_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            storage = root / "storage"
            docs_dir = storage / "docs"
            docs_dir.mkdir(parents=True)
            (docs_dir / "notes.md").write_text("# Notes\n\nWatch the docs cache.\n")

            config = {
                "docs_dir": str(docs_dir),
                "chunk_words": 40,
                "chunk_overlap": 5,
            }
            store = LabStore(storage)

            snapshot = next(iter_document_watch(config, store=store, once=True))

            self.assertTrue(snapshot["changed"])
            self.assertTrue(snapshot["cache_hit"] is False or snapshot["cache_hit"] is True)
            self.assertTrue((storage / "watch" / "latest.json").exists())


if __name__ == "__main__":
    unittest.main()
