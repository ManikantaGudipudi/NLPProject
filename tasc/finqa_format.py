"""Format FinQA JSON examples as a single prompt-friendly context string."""

from __future__ import annotations

from pathlib import Path


def default_finqa_json_path(repo_root: Path, split: str = "dev") -> Path:
    """
    Default path to ``FinQA-main/dataset/<split>.json`` under the project root
    (parent of ``tasc/``).
    ``split`` is the basename without ``.json`` (e.g. ``train``, ``dev``, ``test``).
    """
    return repo_root / "FinQA-main" / "dataset" / f"{split}.json"


def table_to_markdown(table: list[list[str]]) -> str:
    if not table:
        return "(empty table)"
    lines = []
    for row in table:
        lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |")
    return "\n".join(lines)


def format_finqa_context(example: dict) -> str:
    """
    Build hybrid text + table block for LLM consumption.
    """
    pre = "\n".join(example.get("pre_text") or [])
    post = "\n".join(example.get("post_text") or [])
    tab = table_to_markdown(example.get("table") or [])
    q = (example.get("qa") or {}).get("question", "")
    parts = [
        "## Passage (before table)\n" + pre,
        "## Table\n" + tab,
        "## Passage (after table)\n" + post,
        "## Question\n" + q,
    ]
    return "\n\n".join(parts)
