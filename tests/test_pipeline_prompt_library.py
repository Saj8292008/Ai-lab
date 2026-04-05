import unittest

from ai_lab.pipeline.chunking import Chunk
from ai_lab.pipeline.evals import EvalTask
from ai_lab.pipeline.prompts import PromptLibrary
from ai_lab.pipeline.retrieval import SearchHit


class PromptLibraryTests(unittest.TestCase):
    def test_render_includes_profile_rubric_and_context(self) -> None:
        library = PromptLibrary.default()
        task = EvalTask(
            task_id="lab-products",
            prompt="What are the first three lab products to test?",
            reference_answer="The first three lab products to test are ...",
            expected_terms=("private business assistant lora",),
            rubric={"required_terms": ["private business assistant lora"], "pass_threshold": 0.7},
            metadata={"tag": "roadmap"},
        )
        rendered = library.render(
            task=task,
            retrieved=[
                SearchHit(
                    chunk=Chunk(
                        chunk_id="chunk-1",
                        doc_id="doc-1",
                        path="storage/docs/roadmap.md",
                        title="Roadmap",
                        text="The first product is a private business assistant LoRA.",
                        ordinal=0,
                        metadata={},
                    ),
                    score=0.9,
                    matched_terms=("private",),
                )
            ],
            variant_name="rag",
            model_system_prompt="Model prompt",
            variant_instruction="Keep it brief.",
            prompt_profile="rag",
        )

        self.assertIn("Task: What are the first three lab products to test?", rendered.prompt)
        self.assertIn("Workflow profile: rag", rendered.prompt)
        self.assertIn("Retrieved context:", rendered.prompt)
        self.assertIn("private business assistant LoRA", rendered.prompt)
        self.assertIn("Model prompt", rendered.system_prompt)


if __name__ == "__main__":
    unittest.main()
