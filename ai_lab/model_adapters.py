from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class GenerationResult:
    text: str
    provider: str
    model_name: str
    fallback_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class LocalModelAdapter(Protocol):
    provider: str
    model_name: str

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> GenerationResult:
        ...


@dataclass
class HeuristicAdapter:
    provider: str = "heuristic"
    model_name: str = "local-rule-based"

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> GenerationResult:
        context = _extract_block(prompt, "Retrieved context")
        task = _extract_task(prompt)
        answer = _heuristic_answer(task=task, context=context, prompt=prompt)
        return GenerationResult(
            text=answer,
            provider=self.provider,
            model_name=self.model_name,
            fallback_used=False,
            metadata={"temperature": temperature, "max_tokens": max_tokens, "system_prompt": system_prompt or ""},
        )


@dataclass
class OllamaAdapter:
    model_name: str
    host: str = "http://localhost:11434"
    provider: str = "ollama"

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> GenerationResult:
        payload = {
            "model": self.model_name,
            "messages": _build_messages(prompt, system_prompt),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        request = urllib.request.Request(
            f"{self.host.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        text = data.get("message", {}).get("content", "")
        return GenerationResult(
            text=text,
            provider=self.provider,
            model_name=self.model_name,
            fallback_used=False,
            metadata={"host": self.host, "raw": data},
        )


@dataclass
class LlamaCppAdapter:
    model_name: str
    command: list[str]
    provider: str = "llamacpp"

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> GenerationResult:
        if not self.command:
            raise RuntimeError("llama.cpp command is not configured")

        args = [_format_placeholder(part, prompt, system_prompt) for part in self.command]
        env = os.environ.copy()
        env.setdefault("LLAMA_ARG_TEMP", str(temperature))
        env.setdefault("LLAMA_ARG_N_PREDICT", str(max_tokens))
        completed = subprocess.run(
            args,
            input=None,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"llama.cpp command failed with exit code {completed.returncode}: {completed.stderr.strip()}"
            )

        text = completed.stdout.strip()
        return GenerationResult(
            text=text,
            provider=self.provider,
            model_name=self.model_name,
            fallback_used=False,
            metadata={"command": args, "stderr": completed.stderr.strip()},
        )


@dataclass
class AutoAdapter:
    model_name: str
    ollama_host: str = "http://localhost:11434"
    llama_command: list[str] | None = None
    provider: str = "auto"
    _selected: LocalModelAdapter | None = field(default=None, init=False, repr=False)

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> GenerationResult:
        adapter = self._selected or self._resolve_adapter()
        try:
            result = adapter.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self._selected = adapter
            return result
        except Exception as exc:
            fallback = HeuristicAdapter()
            result = fallback.generate(
                prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return GenerationResult(
                text=result.text,
                provider=fallback.provider,
                model_name=fallback.model_name,
                fallback_used=True,
                metadata={"error": str(exc), **result.metadata},
            )

    def _resolve_adapter(self) -> LocalModelAdapter:
        provider = (self.provider or "auto").strip().lower()
        if provider == "heuristic":
            return HeuristicAdapter()
        if provider == "ollama":
            return OllamaAdapter(model_name=self.model_name, host=self.ollama_host)
        if provider == "llamacpp":
            return LlamaCppAdapter(
                model_name=self.model_name,
                command=self.llama_command or _default_llama_command(self.model_name),
            )

        if _ollama_responds(self.ollama_host):
            return OllamaAdapter(model_name=self.model_name, host=self.ollama_host)

        if self.llama_command or _has_llama_binary():
            return LlamaCppAdapter(
                model_name=self.model_name,
                command=self.llama_command or _default_llama_command(self.model_name),
            )

        return HeuristicAdapter()


def resolve_model_adapter(config: dict[str, Any]) -> LocalModelAdapter:
    model_config = config.get("model", {}) or {}
    provider = str(model_config.get("provider", "auto")).strip().lower()
    model_name = str(model_config.get("name", config.get("base_model", "unknown")))
    ollama_host = str(model_config.get("ollama_host", os.environ.get("OLLAMA_HOST", "http://localhost:11434")))
    llama_command = model_config.get("llama_command")
    if isinstance(llama_command, str):
        llama_command = llama_command.split()
    if not isinstance(llama_command, list):
        llama_command = None

    return AutoAdapter(
        model_name=model_name,
        ollama_host=ollama_host,
        llama_command=llama_command,
        provider=provider,
    )


def _build_messages(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _format_placeholder(text: str, prompt: str, system_prompt: str | None) -> str:
    return text.replace("{prompt}", prompt).replace("{system_prompt}", system_prompt or "")


def _extract_block(prompt: str, header: str) -> str:
    lines = prompt.splitlines()
    collecting = False
    inside_block = False
    collected: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower() == f"{header.lower()}:":
            collecting = True
            continue
        if collecting and stripped.startswith("==="):
            if inside_block:
                break
            inside_block = True
            continue
        if collecting:
            collected.append(line)
    return "\n".join(collected).strip()


def _extract_task(prompt: str) -> str:
    for prefix in ("Task:", "Question:", "Prompt:"):
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith(prefix.lower()):
                return stripped.split(":", 1)[1].strip()
    return prompt.strip()


def _heuristic_answer(task: str, context: str, prompt: str) -> str:
    if not context:
        return f"I do not have enough evidence to answer: {task}"

    snippets: list[str] = []
    for line in context.splitlines():
        cleaned = line.strip().lstrip("-*0123456789. ").strip()
        if not cleaned:
            continue
        snippets.append(cleaned)
        if len(snippets) >= 3:
            break

    if not snippets:
        snippets = [context.splitlines()[0].strip()]

    if task and task not in prompt:
        return " ".join(snippets)

    return " ".join(snippets)


def _has_llama_binary() -> bool:
    return shutil.which("llama-cli") is not None or shutil.which("llama") is not None


def _default_llama_command(model_name: str) -> list[str]:
    if shutil.which("llama-cli"):
        return ["llama-cli", "-m", model_name, "-p", "{prompt}"]
    if shutil.which("llama"):
        return ["llama", "-m", model_name, "-p", "{prompt}"]
    return []


def _ollama_responds(host: str) -> bool:
    try:
        request = urllib.request.Request(f"{host.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status == 200
    except Exception:
        return False
