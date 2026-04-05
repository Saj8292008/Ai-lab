from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_lab.paths import config_root, repo_root


def parse_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(0, data)]

    for raw_line in path.read_text().splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        while len(stack) > 1 and indent < stack[-1][0]:
            stack.pop()

        current = stack[-1][1]

        if line.startswith("- "):
            if not isinstance(current, list):
                raise ValueError(f"List item found outside list context: {line}")
            current.append(_parse_scalar(line[2:].strip()))
            continue

        key, value = [part.strip() for part in line.split(":", 1)]
        if value == "":
            next_container: Any = [] if key in {"notes", "tags", "target_modules"} else {}
            current[key] = next_container
            stack.append((indent + 2, next_container))
            continue

        current[key] = _parse_scalar(value)

    return data


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None

    if value.isdigit():
        return int(value)

    try:
        return float(value)
    except ValueError:
        return value


def load_lab_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else config_root() / "lab.sample.yaml"
    if not path.is_absolute():
        path = repo_root() / path
    return parse_simple_yaml(path)

