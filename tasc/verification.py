"""
TASC verification loop — logic and traceability checks.

Public API used by the experiment runner:

    check_logic(question, context, reasoning, llm_fn)       -> dict
    check_traceability(question, context, reasoning, llm_fn) -> dict
    classify_error(logic_result, traceability_result)        -> str
    build_refinement_feedback(logic_result, traceability_result) -> str
"""

from __future__ import annotations

import re
from typing import Any, Callable

import verification_prompts

_VERDICT_RE = re.compile(r"\bVERDICT\b\s*(?::|-)?\s*(PASS|FAIL)\b", re.IGNORECASE)
_REASON_RE = re.compile(r"\bREASON\b\s*(?::|-)?\s*(.+)", re.IGNORECASE | re.DOTALL)


# ------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------

def parse_critic_verdict(raw: str) -> tuple[bool, str]:
    """
    Extract ``VERDICT: PASS/FAIL`` and ``REASON: ...`` from a critic reply.

    Uses the *last* VERDICT line (the model may echo examples in step text).
    Returns ``(passed, reason)``.  Defaults to *(False, "")* when the reply
    cannot be parsed (conservative — treat unparseable as a failure).
    """
    raw = (raw or "").strip()
    passed = False
    reason = ""

    # Normalize line endings and strip common markdown emphasis wrappers so
    # verdict/reason parsing is robust across styles like:
    #   **VERDICT:** PASS
    #   __REASON__: ...
    norm = raw.replace("\r\n", "\n")
    parse_text = re.sub(r"(\*\*|__|`)", "", norm)

    verdicts = list(_VERDICT_RE.finditer(parse_text))
    if verdicts:
        passed = verdicts[-1].group(1).upper() == "PASS"
    else:
        # Fallback: handle odd line-break variants like:
        # "VERDICT\nPASS" or "VERDICT\n\nFAIL"
        verdict_lines = list(re.finditer(r"\bVERDICT\b", parse_text, flags=re.IGNORECASE))
        if verdict_lines:
            tail = parse_text[verdict_lines[-1].end(): verdict_lines[-1].end() + 80]
            m = re.search(r"\b(PASS|FAIL)\b", tail, flags=re.IGNORECASE)
            if m:
                passed = m.group(1).upper() == "PASS"

    reasons = list(_REASON_RE.finditer(parse_text))
    if reasons:
        reason = reasons[-1].group(1).strip().split("\n\n")[0].strip()
        # Trim trailing markdown emphasis artifacts.
        reason = reason.strip("*").strip()

    if not reason and raw:
        reason = raw[:300]

    return passed, reason


# ------------------------------------------------------------------
# Individual checks
# ------------------------------------------------------------------

def check_logic(
    question: str,
    context: str,
    reasoning: str,
    llm_fn: Callable[[str, str], str],
) -> dict[str, Any]:
    """
    f(A) check — is the mathematical approach correct for the question?

    *llm_fn(system_prompt, user_message) -> raw_text*
    """
    if not reasoning or not reasoning.strip():
        return {
            "check": "logic_fA",
            "passed": False,
            "reason": "No reasoning provided by model.",
            "raw_critic": "",
        }

    user_msg = verification_prompts.LOGIC_CHECK_USER.format(
        context=context,
        question=question,
        reasoning=reasoning,
    )
    raw = llm_fn(verification_prompts.LOGIC_CHECK_SYSTEM, user_msg)
    passed, reason = parse_critic_verdict(raw)
    return {
        "check": "logic_fA",
        "passed": passed,
        "reason": reason,
        "raw_critic": raw,
    }


def check_traceability(
    question: str,
    context: str,
    reasoning: str,
    llm_fn: Callable[[str, str], str],
) -> dict[str, Any]:
    """
    A = C check — are the numeric inputs traceable to the source context?

    *llm_fn(system_prompt, user_message) -> raw_text*
    """
    if not reasoning or not reasoning.strip():
        return {
            "check": "traceability_A_eq_C",
            "passed": False,
            "reason": "No reasoning provided by model.",
            "raw_critic": "",
        }

    user_msg = verification_prompts.TRACEABILITY_CHECK_USER.format(
        context=context,
        question=question,
        reasoning=reasoning,
    )
    raw = llm_fn(verification_prompts.TRACEABILITY_CHECK_SYSTEM, user_msg)
    passed, reason = parse_critic_verdict(raw)
    return {
        "check": "traceability_A_eq_C",
        "passed": passed,
        "reason": reason,
        "raw_critic": raw,
    }


# ------------------------------------------------------------------
# Classification and feedback
# ------------------------------------------------------------------

CATEGORY_CORRECT = "correct"
CATEGORY_FA_INCORRECT = "f_A_incorrect"
CATEGORY_A_NEQ_C = "A_neq_C"
CATEGORY_BOTH = "both_errors"


def classify_error(
    logic_result: dict[str, Any],
    traceability_result: dict[str, Any],
) -> str:
    """
    Categorise the reasoning into one of four buckets:

    * ``correct``       — both checks pass
    * ``f_A_incorrect`` — logic fails, traceability passes
    * ``A_neq_C``       — traceability fails, logic passes
    * ``both_errors``   — both checks fail
    """
    logic_ok = logic_result.get("passed", False)
    trace_ok = traceability_result.get("passed", False)

    if logic_ok and trace_ok:
        return CATEGORY_CORRECT
    if not logic_ok and trace_ok:
        return CATEGORY_FA_INCORRECT
    if logic_ok and not trace_ok:
        return CATEGORY_A_NEQ_C
    return CATEGORY_BOTH


def build_refinement_feedback(
    logic_result: dict[str, Any],
    traceability_result: dict[str, Any],
) -> str:
    """
    Human-readable feedback string to inject into the refinement prompt
    (used only by *react_tool_verify*).
    """
    parts: list[str] = []
    if not logic_result.get("passed", False):
        parts.append(
            f"LOGIC ERROR (wrong formula / operation): {logic_result.get('reason', 'unknown')}"
        )
    if not traceability_result.get("passed", False):
        parts.append(
            f"VALUE ERROR (numbers not from source or wrong row/year): "
            f"{traceability_result.get('reason', 'unknown')}"
        )
    return "\n".join(parts) if parts else "No issues detected."
