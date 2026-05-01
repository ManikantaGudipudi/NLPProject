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
    text = (text or "").strip().replace(",", "").replace("$", "")
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

    Gold answers in FinQA are display-rounded from financial reports, so the
    model's more-precise answer (e.g. ``33.33%`` vs gold ``33.3%``) should still
    count.  We use a tiered tolerance:

    * ``rel_tol=5e-3`` (0.5 %) covers one-extra-decimal-place rounding
      (``3.2%`` vs ``3.23%``, ``87%`` vs ``86.64%``, ``104.85%`` vs ``105%``).
    * ``abs_tol=0.005`` catches near-zero values where relative tolerance blows
      up (e.g. ``0%`` vs ``0.001``).
    """
    return math.isclose(pf, gf, rel_tol=5e-3, abs_tol=5e-3)


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


def _strip_pct(s: str) -> str:
    """Remove a trailing ``%`` and surrounding whitespace."""
    return s.replace("%", "").strip()


def answer_matches_qa_answer(gold_answer: str | None, predicted_raw: str | None) -> bool:
    """
    Compare prediction to FinQA ``qa.answer`` (display string: decimals, ``93.5%``, etc.).

    Handles three tricky cases that frequently arise with LLMs:

    1. **Normal path** — both sides go through ``finqa_str_to_num`` (``%`` → ÷100)
       and are compared with ``_finqa_qa_numeric_close``.
    2. **Percent ↔ decimal mismatch** — the model outputs the decimal equivalent
       of a percentage answer (``0.6039`` for gold ``60.3%``), or vice-versa.
       We try ``pred × 100`` vs gold-display, and ``pred`` vs ``gold-display / 100``.
    3. **Missing ``%`` sign** — model writes ``37.81`` when gold is ``37.81%``.
       We compare the raw display numbers (``37.81`` ≈ ``37.81``) directly.
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

    # --- canonical comparison (both through finqa_str_to_num) ---
    gf = finqa_str_to_num(gold_answer)
    pf = finqa_str_to_num(pred)
    if gf is not None and pf is not None:
        if _finqa_qa_numeric_close(gf, pf):
            return True

    # --- cross-format: missing % sign ---
    # gold = "37.81%", pred = "37.81" → compare display numbers directly
    gold_has_pct = "%" in gold_answer
    pred_has_pct = "%" in pred
    if gold_has_pct != pred_has_pct:
        g_display = _to_float(_strip_pct(gold_answer))
        p_display = _to_float(_strip_pct(pred))
        if g_display is not None and p_display is not None:
            if _finqa_qa_numeric_close(g_display, p_display):
                return True

    # --- cross-format: decimal ↔ percentage ---
    # gold = "60.3%" (gf=0.603), pred = "0.6039" (pf=0.6039) → already close
    # gold = "57%" (gf=0.57), pred = "57" (pf=57) → try pf/100
    if gf is not None and pf is not None:
        if pf != 0 and _finqa_qa_numeric_close(gf, pf / 100.0):
            return True
        if gf != 0 and _finqa_qa_numeric_close(gf * 100.0, pf):
            return True

    return pl == gold_answer.lower().replace(",", "").strip()


def extract_react_final_answer(predicted_raw: str | None) -> str | None:
    """
    Last ``Final Answer:`` line (single-line value), or ``None``.

    Cleans common LLM noise: leading ``$``, inline calculator expressions
    (``calculator (1+2)= 3`` → ``3``), units, etc.
    """
    if predicted_raw is None or not str(predicted_raw).strip():
        return None
    matches = list(_REACT_FINAL_RE.finditer(str(predicted_raw).strip()))
    if not matches:
        return None
    val = matches[-1].group(1).strip()

    # Model sometimes writes "calculator <expr> = <result>" as final answer
    eq_parts = val.rsplit("=", 1)
    if len(eq_parts) == 2 and eq_parts[1].strip():
        candidate = eq_parts[1].strip()
        if re.match(r"^-?\d", candidate):
            val = candidate

    val = val.replace("$", "").replace(",", "").strip()
    # Strip trailing unit words like "million", "billion", "thousand"
    val = re.sub(r"\s*(million|billion|thousand|mn|bn)s?\s*$", "", val, flags=re.IGNORECASE)
    return val.strip() if val.strip() else None


def answer_matches_react_qa_answer(gold_answer: str | None, predicted_raw: str | None) -> bool:
    """
    Like ``answer_matches_qa_answer`` but reads the last ``Final Answer:`` line (ReAct).
    """
    tail = extract_react_final_answer(predicted_raw)
    if tail is None:
        return False
    return answer_matches_qa_answer(gold_answer, f"FINAL: {tail}")


def _normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("’", "'")
    s = re.sub(r"[^a-z0-9%]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_num_loose(text: str) -> float | None:
    """Parse numeric-like strings for TAT-QA without implicit % -> /100 conversion."""
    if text is None:
        return None
    t = str(text).strip().lower()
    if not t:
        return None
    t = t.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        return float(t)
    except ValueError:
        return None


def _split_span_answers(text: str) -> list[str]:
    # Primary canonical separator used by this project for multi-span golds.
    parts = [p.strip() for p in str(text).split("|")]
    return [p for p in parts if p]


def _normalize_span_item(s: str) -> str:
    """Normalize span labels while tolerating trailing type/category wording."""
    t = _normalize_text(s)
    # Common TAT-QA label tails that are often omitted by models.
    t = re.sub(r"\b(type|types|contract|contracts)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def answer_matches_tatqa(
    gold_answer: str | None,
    predicted_raw: str | None,
    *,
    answer_type: str | None = None,
    scale: str | None = None,
) -> bool:
    """
    TAT-QA-aware answer matcher.

    Supports:
      - span / multi-span textual answers
      - numeric / arithmetic answers with optional $, commas, and % formatting
    """
    if gold_answer is None or predicted_raw is None:
        return False

    pred = parse_predicted_answer(predicted_raw)
    if pred is None:
        return False

    gold = str(gold_answer).strip()
    pred = str(pred).strip()
    if not gold:
        return False

    at = (answer_type or "").strip().lower()
    sc = (scale or "").strip().lower()

    # Numeric span answers are common in TAT-QA; handle them numerically first.
    gf_num = _parse_num_loose(gold)
    pf_num = _parse_num_loose(pred)
    if gf_num is not None and pf_num is not None:
        if math.isclose(gf_num, pf_num, rel_tol=5e-3, abs_tol=5e-3):
            return True
        if sc == "percent":
            if math.isclose(gf_num, pf_num * 100.0, rel_tol=5e-3, abs_tol=5e-3):
                return True
            if math.isclose(gf_num * 100.0, pf_num, rel_tol=5e-3, abs_tol=5e-3):
                return True

    # --- Textual answers (span / multi-span) ---
    if at in {"span", "multi-span"}:
        gold_items = _split_span_answers(gold)
        pred_norm = _normalize_text(pred)
        if not gold_items:
            return False
        gold_norm = [_normalize_text(g) for g in gold_items]
        pred_relaxed = _normalize_span_item(pred)
        gold_relaxed = [_normalize_span_item(g) for g in gold_items]
        if at == "span":
            g = gold_norm[0]
            # allow exact or containment for slightly verbose responses
            if pred_norm == g or g in pred_norm or pred_norm in g:
                return True
            if gold_relaxed and gold_relaxed[0]:
                gr = gold_relaxed[0]
                return pred_relaxed == gr or gr in pred_relaxed or pred_relaxed in gr

        # multi-span: each expected span must appear in prediction
        if all(g and g in pred_norm for g in gold_norm):
            return True
        return all(g and g in pred_relaxed for g in gold_relaxed)

    # --- Numeric-like answers ---
    gf = gf_num
    pf = pf_num
    if gf is not None and pf is not None:
        # Standard numeric closeness
        if math.isclose(gf, pf, rel_tol=5e-3, abs_tol=5e-3):
            return True
        # For percent-scaled questions, tolerate decimal vs percent forms.
        if sc == "percent":
            if math.isclose(gf, pf * 100.0, rel_tol=5e-3, abs_tol=5e-3):
                return True
            if math.isclose(gf * 100.0, pf, rel_tol=5e-3, abs_tol=5e-3):
                return True
        return False

    # Fallback textual compare
    return _normalize_text(gold) == _normalize_text(pred)


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
