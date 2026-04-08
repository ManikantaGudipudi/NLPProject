"""
Parse ReAct-style model output: Thought, Action, Action Input, Final Answer.

Tolerant of minor spacing/case; prefers line-oriented structure.
"""

from __future__ import annotations

import re
from typing import Any


# Line-start keywords (case-insensitive for labels)
_THOUGHT = re.compile(r"(?im)^Thought:\s*(.*?)(?=^(?:Action|Action Input|Observation|Final Answer):|\Z)", re.DOTALL)
_ACTION = re.compile(r"(?im)^Action:\s*(\S.*?)\s*$")
_ACTION_INPUT = re.compile(
    r"(?im)^Action Input:\s*(.*?)(?=^(?:Observation|Thought|Action|Final Answer):|\Z)",
    re.DOTALL,
)
_FINAL_ANSWER = re.compile(r"(?im)^Final Answer:\s*([^\n]+?)\s*$")
_OBSERVATION = re.compile(r"(?im)^Observation:\s*([^\n]+?)\s*$")


def parse_react_block(text: str) -> dict[str, Any]:
    """
    Extract fields from one model turn. Missing fields are ``None``.

    Keys: ``thought``, ``action``, ``action_input``, ``observation``, ``final_answer``
    (``final_answer`` / ``observation`` use the last matching line if repeated).
    """
    raw = (text or "").strip()
    out: dict[str, Any] = {
        "thought": None,
        "action": None,
        "action_input": None,
        "observation": None,
        "final_answer": None,
        "raw": raw,
    }

    fa = list(_FINAL_ANSWER.finditer(raw))
    if fa:
        out["final_answer"] = _clean_value(fa[-1].group(1))

    obs = list(_OBSERVATION.finditer(raw))
    if obs:
        out["observation"] = obs[-1].group(1).strip()

    m = _THOUGHT.search(raw)
    if m:
        out["thought"] = m.group(1).strip()

    m = _ACTION.search(raw)
    if m:
        out["action"] = m.group(1).strip().lower()

    m = _ACTION_INPUT.search(raw)
    if m:
        # Often one line; allow continuation without new keywords
        ai = m.group(1).strip()
        first_line = ai.split("\n")[0].strip()
        out["action_input"] = first_line

    return out


def _clean_value(s: str) -> str:
    s = s.strip()
    # Single-line final answer for scoring
    line = s.split("\n")[0].strip()
    return line


def wants_calculator(parsed: dict[str, Any]) -> bool:
    """True if model requested the calculator action with non-empty input."""
    act = parsed.get("action") or ""
    inp = parsed.get("action_input") or ""
    return act == "calculator" and bool(str(inp).strip())


def parse_observation_line(text: str) -> str | None:
    """Last ``Observation:`` line (single-line payload), or ``None``."""
    raw = (text or "").strip()
    matches = list(_OBSERVATION.finditer(raw))
    return matches[-1].group(1).strip() if matches else None
