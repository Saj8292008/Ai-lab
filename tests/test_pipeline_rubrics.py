import unittest

from ai_lab.pipeline.evals import EvalTask, score_answer
from ai_lab.pipeline.chunking import Chunk
from ai_lab.pipeline.retrieval import SearchHit


class RubricScoringTests(unittest.TestCase):
    def test_task_specific_rubric_passes_on_required_terms(self) -> None:
        task = EvalTask(
            task_id="products",
            prompt="What are the first three lab products to test?",
            reference_answer="The first three lab products to test are a private business assistant LoRA, a niche creator style/model pack, and a local knowledge worker copilot.",
            expected_terms=("private business assistant lora", "niche creator style/model pack", "local knowledge worker copilot"),
            rubric={
                "required_terms": [
                    "private business assistant lora",
                    "niche creator style/model pack",
                    "local knowledge worker copilot",
                ],
                "must_use_retrieval": True,
                "required_coverage": 0.67,
                "pass_threshold": 0.68,
                "min_retrieved_docs": 1,
            },
        )
        metrics = score_answer(
            task,
            "The first three lab products to test are a private business assistant LoRA, a niche creator style/model pack, and a local knowledge worker copilot for docs and drafting.",
            retrieved_hits=[
                SearchHit(
                    chunk=Chunk(
                        chunk_id="chunk-1",
                        doc_id="doc-1",
                        path="storage/docs/product-roadmap.md",
                        title="Product Roadmap Notes",
                        text="The first useful products for this lab are a private business assistant LoRA, a niche creator style/model pack, and a local knowledge worker copilot for docs and drafting.",
                        ordinal=0,
                        metadata={},
                    ),
                    score=0.9,
                    matched_terms=("products",),
                )
            ],
        )

        self.assertTrue(metrics["passed"])
        self.assertGreater(metrics["score"], 0.6)

    def test_task_specific_rubric_rejects_forbidden_language(self) -> None:
        task = EvalTask(
            task_id="cadence",
            prompt="What is the recommended research cadence?",
            reference_answer="Daily means inspect data, run one controlled experiment, and log one finding.",
            expected_terms=("daily", "weekly", "monthly"),
            rubric={
                "required_terms": ["daily", "weekly", "monthly"],
                "forbidden_terms": ["from scratch"],
                "required_coverage": 0.5,
                "pass_threshold": 0.5,
            },
        )
        metrics = score_answer(task, "The lab should train from scratch and do nothing else.", [])

        self.assertFalse(metrics["passed"])
        self.assertEqual(metrics["forbidden_penalty"], 1.0)


if __name__ == "__main__":
    unittest.main()
