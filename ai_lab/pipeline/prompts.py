from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ai_lab.pipeline.evals import EvalTask
from ai_lab.pipeline.retrieval import SearchHit


@dataclass(frozen=True)
class PromptProfile:
    name: str
    system_prompt: str = ""
    instruction: str = ""
    retrieval_instruction: str = ""
    response_format: str = "Return 1-3 concise sentences with the answer only."
    use_retrieval: bool = True
    include_citations: bool = False


@dataclass(frozen=True)
class RenderedPrompt:
    profile_name: str
    system_prompt: str
    prompt: str


@dataclass
class PromptLibrary:
    profiles: dict[str, PromptProfile] = field(default_factory=dict)
    default_profile_name: str = "rag"

    @classmethod
    def default(cls) -> "PromptLibrary":
        return cls(
            profiles={
                "baseline": PromptProfile(
                    name="baseline",
                    system_prompt="You are a concise local research assistant.",
                    instruction="Answer from the question alone and be explicit when evidence is missing.",
                    retrieval_instruction="Do not rely on retrieved context.",
                    response_format="Return 1-3 concise sentences with the answer only.",
                    use_retrieval=False,
                ),
                "rag": PromptProfile(
                    name="rag",
                    system_prompt="You are a local research assistant. Use the retrieved context and prefer direct evidence.",
                    instruction="Use the retrieved context, prefer direct evidence, and keep the answer concise.",
                    retrieval_instruction="Ground the answer in the strongest retrieved evidence.",
                    response_format="Return 1-3 concise sentences with the answer only.",
                    use_retrieval=True,
                ),
                "synthesis": PromptProfile(
                    name="synthesis",
                    system_prompt="You synthesize evidence across chunks without overclaiming.",
                    instruction="Combine the best evidence across retrieved chunks and highlight the most relevant signal.",
                    retrieval_instruction="Synthesize multiple snippets when they reinforce the same answer.",
                    response_format="Return a short synthesis with the answer only.",
                    use_retrieval=True,
                    include_citations=True,
                ),
            }
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PromptLibrary":
        prompt_config = config.get("prompt_library", {}) or {}
        if not isinstance(prompt_config, dict):
            return cls.default()

        profiles: dict[str, PromptProfile] = {}
        raw_profiles = prompt_config.get("profiles", {}) or {}
        if isinstance(raw_profiles, dict):
            for profile_name, payload in raw_profiles.items():
                if not isinstance(payload, dict):
                    continue
                profiles[str(profile_name)] = PromptProfile(
                    name=str(profile_name),
                    system_prompt=str(payload.get("system_prompt", "")).strip(),
                    instruction=str(payload.get("instruction", "")).strip(),
                    retrieval_instruction=str(payload.get("retrieval_instruction", "")).strip(),
                    response_format=str(
                        payload.get("response_format", "Return 1-3 concise sentences with the answer only.")
                    ).strip(),
                    use_retrieval=bool(payload.get("use_retrieval", True)),
                    include_citations=bool(payload.get("include_citations", False)),
                )

        if not profiles:
            profiles = cls.default().profiles

        return cls(
            profiles=profiles,
            default_profile_name=str(prompt_config.get("default_profile", "rag")).strip() or "rag",
        )

    def profile_for_variant(self, variant_name: str, fallback_name: str | None = None) -> PromptProfile:
        candidates = [variant_name, fallback_name, self.default_profile_name, "rag", "baseline"]
        for name in candidates:
            if name and name in self.profiles:
                return self.profiles[name]
        return next(iter(self.profiles.values()))

    def render(
        self,
        *,
        task: EvalTask,
        retrieved: list[SearchHit],
        variant_name: str,
        model_system_prompt: str = "",
        variant_instruction: str = "",
        prompt_profile: str | None = None,
    ) -> RenderedPrompt:
        profile = self.profile_for_variant(prompt_profile or variant_name)
        system_prompt = _join_parts(model_system_prompt, profile.system_prompt, variant_instruction)
        prompt = _join_parts(
            f"Task: {task.prompt}",
            f"Workflow profile: {profile.name}",
            _task_context(task),
            _rubric_block(task),
            profile.instruction,
            _retrieval_block(retrieved, profile),
            profile.response_format,
        )
        return RenderedPrompt(profile_name=profile.name, system_prompt=system_prompt, prompt=prompt)


def _join_parts(*parts: str) -> str:
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return "\n\n".join(cleaned)


def _task_context(task: EvalTask) -> str:
    workflow = str(task.metadata.get("tag", "")).strip()
    source_ids = ", ".join(task.source_doc_ids) if task.source_doc_ids else "unknown"
    lines = [
        f"Task id: {task.task_id}",
        f"Workflow: {workflow or 'general'}",
        f"Source docs: {source_ids}",
    ]
    return "\n".join(lines)


def _rubric_block(task: EvalTask) -> str:
    rubric = task.rubric or {}
    required_terms = ", ".join(str(term) for term in rubric.get("required_terms", task.expected_terms)) or "none"
    forbidden_terms = ", ".join(str(term) for term in rubric.get("forbidden_terms", [])) or "none"
    lines = [
        "Rubric:",
        f"- Required terms: {required_terms}",
        f"- Forbidden terms: {forbidden_terms}",
        f"- Must use retrieval: {bool(rubric.get('must_use_retrieval', False))}",
        f"- Pass threshold: {float(rubric.get('pass_threshold', 0.0)):.2f}",
    ]
    return "\n".join(lines)


def _retrieval_block(retrieved: list[SearchHit], profile: PromptProfile) -> str:
    lines = ["Retrieved context:", "==="]
    if retrieved:
        for index, hit in enumerate(retrieved, start=1):
            lines.append(f"[{index}] {hit.chunk.title} ({hit.chunk.path}) :: {hit.chunk.doc_id}")
            lines.append(hit.chunk.text.strip())
            if profile.include_citations:
                lines.append(f"Citation: [{hit.chunk.doc_id}:{hit.chunk.ordinal}]")
            lines.append("---")
    else:
        lines.append("(no retrieved context)")
    lines.append("===")

    if profile.retrieval_instruction:
        lines.extend(["Retrieval guidance:", profile.retrieval_instruction])

    return "\n".join(lines)
