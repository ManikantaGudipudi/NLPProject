"""Parse model outputs and compare to FinQA gold (qa.answer or exe_ans)."""

from __future__ import annotations

import math
import re
from typing import Any


_FINAL_RE = re.compile(r"^\s*FINAL:\s*(.+)\s*$", re.IGNORECASE | re.MULTILINE)
_REACT_FINAL_RE = re.compile(r"(?im)^Final Answer:\s*([^\n]+?)\s*$")


def finqa_str_to_num(text: str) -> float | None:
    """
    Parse a FinQA-style display string (aligned with FinQA evaluate.py str_to_num
    for floats and simple percentages). Returns None if not parseable as a number.
    """
    text = (text or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        if "%" in text:
            t = text.replace("%", "").strip()
            try:
                return float(t) / 100.0
            except ValueError:
                return None
        return None


def parse_predicted_answer(raw: str) -> str | None:
    """
    Prefer the last line 'FINAL: <value>'; else last plausible number; else yes/no.
    Returns a normalized string for comparison, or None if nothing parsed.
    """
    if not raw or not raw.strip():
        return None
    matches = list(_FINAL_RE.finditer(raw))
    if matches:
        val = matches[-1].group(1).strip().strip("`\"'")
        val = val.replace("$", "").replace(",", "").strip()
        low = val.lower()
        if low in ("yes", "no"):
            return low
        return val
    text = raw.strip()
    low = text.lower()
    if re.search(r"\byes\b", low) and not re.search(r"\bno\b", low):
        return "yes"
    if re.search(r"\bno\b", low) and not re.search(r"\byes\b", low):
        return "no"
    # Last resort: find floats / ints (prefer last number in text)
    nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:e[+-]?\d+)?", text, re.IGNORECASE)
    if nums:
        return nums[-1].replace(",", "")
    return None


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _finqa_qa_numeric_close(gf: float, pf: float) -> bool:
    """
    Compare FinQA ``qa.answer``-style numbers after ``finqa_str_to_num``.

    ``rel_tol=1e-4`` is too strict for common display rounding (e.g. ``60.3%`` vs
    ``60.31%``, or ``3044%`` vs ``3044.32%``). Slightly looser tolerances match
    one-extra-decimal and similar table/LLM rounding without opening large holes.
    """
    return math.isclose(pf, gf, rel_tol=5e-4, abs_tol=1e-4)


def answer_matches(gold: Any, predicted_raw: str | None) -> bool:
    """
    Gold is FinQA ``exe_ans``: float or occasionally 'yes'/'no'.
    """
    if predicted_raw is None:
        return False
    pred = parse_predicted_answer(predicted_raw)
    if pred is None:
        return False

    if isinstance(gold, str) and gold.lower() in ("yes", "no"):
        return pred.strip().lower() == gold.lower()

    pf = _to_float(pred)
    gf = float(gold) if not isinstance(gold, str) else _to_float(gold)
    if pf is None or gf is None:
        return str(pred).strip().lower() == str(gold).strip().lower()
    return math.isclose(pf, gf, rel_tol=1e-4, abs_tol=1e-5)


def answer_matches_qa_answer(gold_answer: str | None, predicted_raw: str | None) -> bool:
    """
    Compare prediction to FinQA ``qa.answer`` (display string: decimals, ``93.5%``, etc.).
    Yes/no questions: case-insensitive string match.
    Numeric: canonical float via ``finqa_str_to_num`` on both sides, then ``math.isclose``.
    """
    if predicted_raw is None or gold_answer is None:
        return False
    gold_answer = gold_answer.strip()
    if not gold_answer:
        return False

    pred = parse_predicted_answer(predicted_raw)
    if pred is None:
        return False

    gl = gold_answer.lower()
    pl = pred.strip().lower()
    if gl in ("yes", "no"):
        return pl == gl

    gf = finqa_str_to_num(gold_answer)
    pf = finqa_str_to_num(pred)
    if gf is not None and pf is not None:
        return _finqa_qa_numeric_close(gf, pf)

    return pl == gold_answer.lower().replace(",", "").strip()


def extract_react_final_answer(predicted_raw: str | None) -> str | None:
    """Last ``Final Answer:`` line (single-line value), or ``None``."""
    if predicted_raw is None or not str(predicted_raw).strip():
        return None
    matches = list(_REACT_FINAL_RE.finditer(str(predicted_raw).strip()))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def answer_matches_react_qa_answer(gold_answer: str | None, predicted_raw: str | None) -> bool:
    """
    Like ``answer_matches_qa_answer`` but reads the last ``Final Answer:`` line (ReAct).
    """
    tail = extract_react_final_answer(predicted_raw)
    if tail is None:
        return False
    return answer_matches_qa_answer(gold_answer, f"FINAL: {tail}")


def extract_reasoning_before_final(raw: str | None) -> str | None:
    """
    Return text before the last ``FINAL:`` line (model chain-of-thought / work).
    If there is no ``FINAL:`` line, returns the full message (minus leading/trailing
    whitespace). If the reply is only ``FINAL: ...``, returns ``None``.
    """
    if raw is None or not str(raw).strip():
        return None
    raw = str(raw)
    matches = list(_FINAL_RE.finditer(raw))
    if not matches:
        return raw.strip()
    last = matches[-1]
    before = raw[: last.start()].strip()
    return before if before else None
