#!/usr/bin/env python3
"""
TASC unified experiment runner.

Three modes:
  cot              — Chain-of-Thought only.  Verification identifies errors (read-only).
  react_tool       — ReAct + calculator.     Verification identifies errors (read-only).
  react_tool_verify — ReAct + calculator + verification loop that feeds errors back
                      to the LLM for regeneration (up to --max-retries).

Usage (from repo root NLPProject):
  python tasc/scripts/run_experiment.py --mode cot              --limit 5
  python tasc/scripts/run_experiment.py --mode react_tool       --limit 5
  python tasc/scripts/run_experiment.py --mode react_tool_verify --limit 5 --max-retries 3

  # Gemini backend
  export GEMINI_API_KEY=...
  python tasc/scripts/run_experiment.py --mode cot --backend gemini --limit 5

Each run produces two files in tasc/results/:
  <mode>_<backend>_<split>_<ts>.jsonl          — per-query records
  <mode>_<backend>_<split>_<ts>_summary.json   — aggregate statistics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_TASC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TASC_ROOT))

import finqa_format  # noqa: E402
import gemini_client  # noqa: E402
import numeric_eval  # noqa: E402
import ollama_client  # noqa: E402
import react_parse  # noqa: E402
import react_prompts  # noqa: E402
import react_tools  # noqa: E402
import verification  # noqa: E402
import verification_prompts  # noqa: E402


# ======================================================================
# CoT prompts (same prompts used by run_numeric_baseline.py)
# ======================================================================

COT_SYSTEM_PROMPT = """\
You are a precise financial question-answering assistant.

You must answer using only the passage and table provided in the user message.
Do not use outside knowledge.
Do not guess.

First, reason step by step in a concise and explicit way.
Your reasoning must follow this structure:

1. RELEVANT VALUES:
   - List only the values taken from the passage/table that are needed
2. OPERATION:
   - State the formula, comparison, or logic being used
3. COMPUTATION / DECISION:
   - Perform the calculation or yes/no check clearly
4. ANSWER CHECK:
   - Briefly confirm that the final value matches the question type
     (decimal / percentage / yes-no)

Then, on the very last line, output exactly:
FINAL: <value>

Rules for <value>:
- It must be one of:
  - a decimal number
  - a percentage with % sign
  - yes
  - no
- Output only the value after FINAL:
- Do not include $ signs
- Do not include commas
- Do not include units unless the answer is explicitly a percentage
- Ratios between 0 and 1 must be returned as decimals (example: 0.637, not 63.7%)
- If the question asks for a percentage, return a percentage with % sign
- If the question is yes/no, return exactly yes or no

Important:
- The grader reads only the last line beginning with FINAL:
- Therefore the last line must be exactly in the required format
- Do not place any text after the FINAL line
"""

COT_USER_SUFFIX = """

Answer the question using only the passage/table above.

Show concise step-by-step reasoning in this order:
1. RELEVANT VALUES
2. OPERATION
3. COMPUTATION / DECISION
4. ANSWER CHECK

Then end with exactly one final line:
FINAL: <value>
"""


# ======================================================================
# LLM call helpers
# ======================================================================

def _call_llm(
    backend: str,
    system_prompt: str,
    user_msg: str,
    *,
    gemini_model: str | None = None,
    timeout: int = 600,
    max_tokens: int = 2048,
) -> str:
    """Single-turn LLM call (system + user)."""
    if backend == "ollama":
        return ollama_client.ollama_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            timeout=timeout,
            options={
                "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", "0.1")),
                "num_predict": int(os.environ.get("OLLAMA_NUM_PREDICT", str(max_tokens))),
            },
        )
    if backend == "gemini":
        return gemini_client.generate_content(
            system_prompt,
            user_msg,
            model=gemini_model,
            timeout=timeout,
            max_output_tokens=max_tokens,
        )
    raise ValueError(f"Unknown backend {backend!r}")


def _call_llm_chat(
    backend: str,
    system_prompt: str,
    messages: list[dict[str, str]],
    *,
    gemini_model: str | None = None,
    timeout: int = 600,
    max_tokens: int = 2048,
) -> str:
    """Multi-turn LLM call (system + message list)."""
    if backend == "ollama":
        return ollama_client.ollama_chat(
            [{"role": "system", "content": system_prompt}, *messages],
            timeout=timeout,
            options={
                "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", "0.1")),
                "num_predict": int(os.environ.get("OLLAMA_NUM_PREDICT", str(max_tokens))),
            },
        )
    if backend == "gemini":
        return gemini_client.generate_content_chat(
            messages,
            system_instruction=system_prompt,
            model=gemini_model,
            timeout=timeout,
            max_output_tokens=max_tokens,
        )
    raise ValueError(f"Unknown backend {backend!r}")


def _make_critic_fn(backend: str, gemini_model: str | None, timeout: int):
    """Return a ``(system, user) -> str`` callable for the verification module."""
    def fn(system_prompt: str, user_msg: str) -> str:
        return _call_llm(
            backend, system_prompt, user_msg,
            gemini_model=gemini_model,
            timeout=timeout,
            max_tokens=2048,
        )
    return fn


# ======================================================================
# Generation: CoT
# ======================================================================

def generate_cot(
    example: dict,
    *,
    backend: str,
    gemini_model: str | None,
    timeout: int,
) -> dict:
    """
    Run CoT generation.  Returns dict with keys:
    raw, reasoning, predicted, model_calls.
    """
    context = finqa_format.format_finqa_context(example)
    user_msg = context + COT_USER_SUFFIX

    raw = _call_llm(
        backend, COT_SYSTEM_PROMPT, user_msg,
        gemini_model=gemini_model, timeout=timeout, max_tokens=2048,
    )
    predicted = numeric_eval.parse_predicted_answer(raw)
    reasoning = numeric_eval.extract_reasoning_before_final(raw) or ""

    return {
        "raw": raw,
        "reasoning": reasoning,
        "predicted": predicted,
        "model_calls": 1,
        "tool_used": False,
        "observation": None,
    }


# ======================================================================
# Generation: ReAct + calculator (shared by react_tool & react_tool_verify)
# ======================================================================

def generate_react(
    example: dict,
    *,
    backend: str,
    gemini_model: str | None,
    timeout: int,
    debug: bool = False,
    feedback: str | None = None,
) -> dict:
    """
    Run one full ReAct generation (up to 2 LLM calls: initial + post-observation).

    If *feedback* is provided it is prepended to the user message so the model
    is aware of prior verification failures (used by react_tool_verify retries).

    Returns dict with keys:
    raw_turn1, raw_turn2, reasoning, predicted, model_calls,
    tool_used, observation, parsed_turn1.
    """
    context = finqa_format.format_finqa_context(example)
    user_suffix = react_prompts.REACT_USER_SUFFIX

    if feedback:
        user1 = (
            verification_prompts.REFINEMENT_PREFIX.format(feedback=feedback)
            + "\n" + context + user_suffix
        )
    else:
        user1 = context + user_suffix

    messages1: list[dict[str, str]] = [{"role": "user", "content": user1}]
    raw1 = _call_llm_chat(
        backend, react_prompts.REACT_SYSTEM_PROMPT, messages1,
        gemini_model=gemini_model, timeout=timeout, max_tokens=2048,
    )
    p1 = react_parse.parse_react_block(raw1)

    raw2 = ""
    observation: str | None = None
    calls = 1
    tool_used = False

    if react_parse.wants_calculator(p1):
        tool_used = True
        expr = (p1.get("action_input") or "").strip()
        if debug:
            print(f"  DEBUG tool_call: action_input={expr!r}", flush=True)
        val, err = react_tools.safe_calculate(expr)
        if err is not None:
            observation = f"Error: {err}"
        else:
            fv = float(val)
            observation = str(int(fv)) if fv.is_integer() else f"{fv:.12g}"

        user2 = react_prompts.REACT_AFTER_OBSERVATION_USER.format(observation=observation)
        messages2: list[dict[str, str]] = [
            {"role": "user", "content": user1},
            {"role": "assistant", "content": raw1},
            {"role": "user", "content": user2},
        ]
        raw2 = _call_llm_chat(
            backend, react_prompts.REACT_SYSTEM_PROMPT, messages2,
            gemini_model=gemini_model, timeout=timeout, max_tokens=2048,
        )
        calls = 2

    final_text = raw2 if raw2 else raw1
    predicted = numeric_eval.extract_react_final_answer(final_text)

    # Build a combined reasoning string for the critic
    reasoning_parts: list[str] = []
    if p1.get("thought"):
        reasoning_parts.append(f"Thought: {p1['thought']}")
    if p1.get("action_input"):
        reasoning_parts.append(f"Calculation: {p1['action_input']}")
    if observation:
        reasoning_parts.append(f"Observation: {observation}")
    if raw2:
        p2 = react_parse.parse_react_block(raw2)
        if p2.get("thought"):
            reasoning_parts.append(f"Conclusion: {p2['thought']}")
    reasoning = "\n".join(reasoning_parts) if reasoning_parts else raw1

    return {
        "raw_turn1": raw1,
        "raw_turn2": raw2 or None,
        "reasoning": reasoning,
        "predicted": predicted,
        "model_calls": calls,
        "tool_used": tool_used,
        "observation": observation,
        "parsed_turn1": p1,
    }


# ======================================================================
# Verification wrapper
# ======================================================================

def verify_reasoning(
    question: str,
    context: str,
    reasoning: str,
    critic_fn,
) -> dict:
    """
    Run both verification checks and classify the error.

    Returns dict with keys: logic, traceability, category.
    """
    logic_result = verification.check_logic(question, context, reasoning, critic_fn)
    trace_result = verification.check_traceability(question, context, reasoning, critic_fn)
    category = verification.classify_error(logic_result, trace_result)
    return {
        "logic": logic_result,
        "traceability": trace_result,
        "category": category,
    }


# ======================================================================
# Scoring
# ======================================================================

def _score_cot(gold: str, raw: str) -> bool:
    return numeric_eval.answer_matches_qa_answer(gold, raw)


def _score_react(gold: str, final_text: str) -> bool:
    return numeric_eval.answer_matches_react_qa_answer(gold, final_text)


# ======================================================================
# Per-example pipelines
# ======================================================================

def run_cot_pipeline(
    example: dict,
    gold: str,
    *,
    backend: str,
    gemini_model: str | None,
    timeout: int,
    critic_fn,
) -> dict:
    """CoT: generate → score → verify (read-only)."""
    gen = generate_cot(
        example, backend=backend, gemini_model=gemini_model, timeout=timeout,
    )
    answer_correct = _score_cot(gold, gen["raw"])
    context = finqa_format.format_finqa_context(example)
    question = (example.get("qa") or {}).get("question", "")
    verif = verify_reasoning(question, context, gen["reasoning"], critic_fn)
    return {
        **gen,
        "answer_correct": answer_correct,
        "verification": verif,
    }


def run_react_pipeline(
    example: dict,
    gold: str,
    *,
    backend: str,
    gemini_model: str | None,
    timeout: int,
    critic_fn,
    debug: bool = False,
) -> dict:
    """ReAct + tool: generate → score → verify (read-only)."""
    gen = generate_react(
        example, backend=backend, gemini_model=gemini_model,
        timeout=timeout, debug=debug,
    )
    final_text = gen["raw_turn2"] if gen["raw_turn2"] else gen["raw_turn1"]
    answer_correct = _score_react(gold, final_text)
    context = finqa_format.format_finqa_context(example)
    question = (example.get("qa") or {}).get("question", "")
    verif = verify_reasoning(question, context, gen["reasoning"], critic_fn)
    return {
        **gen,
        "answer_correct": answer_correct,
        "verification": verif,
    }


def run_react_verify_pipeline(
    example: dict,
    gold: str,
    *,
    backend: str,
    gemini_model: str | None,
    timeout: int,
    critic_fn,
    max_retries: int = 3,
    debug: bool = False,
) -> dict:
    """
    ReAct + tool + verification loop.

    Generate → verify → (if errors) feed critic feedback → regenerate → …
    Stops when verification passes or *max_retries* is exhausted.
    The last attempt is used for scoring and categorisation.
    """
    context = finqa_format.format_finqa_context(example)
    question = (example.get("qa") or {}).get("question", "")

    # --- initial attempt ---
    gen = generate_react(
        example, backend=backend, gemini_model=gemini_model,
        timeout=timeout, debug=debug,
    )
    verif = verify_reasoning(question, context, gen["reasoning"], critic_fn)

    attempts: list[dict] = []
    attempts.append({
        "attempt": 1,
        "reasoning": gen["reasoning"],
        "predicted": gen.get("predicted"),
        "verification": verif,
        "model_calls": gen["model_calls"],
    })

    total_model_calls = gen["model_calls"] + 2  # +2 for the two critic calls

    # --- retry loop ---
    retry = 0
    while verif["category"] != verification.CATEGORY_CORRECT and retry < max_retries:
        retry += 1
        feedback = verification.build_refinement_feedback(
            verif["logic"], verif["traceability"],
        )
        print(f"    retry {retry}/{max_retries}: {verif['category']} → regenerating", flush=True)

        gen = generate_react(
            example, backend=backend, gemini_model=gemini_model,
            timeout=timeout, debug=debug, feedback=feedback,
        )
        verif = verify_reasoning(question, context, gen["reasoning"], critic_fn)
        total_model_calls += gen["model_calls"] + 2

        attempts.append({
            "attempt": retry + 1,
            "reasoning": gen["reasoning"],
            "predicted": gen.get("predicted"),
            "verification": verif,
            "model_calls": gen["model_calls"],
        })

    # --- score the LAST attempt ---
    final_text = gen["raw_turn2"] if gen["raw_turn2"] else gen["raw_turn1"]
    answer_correct = _score_react(gold, final_text)

    return {
        **gen,
        "answer_correct": answer_correct,
        "verification": verif,
        "retries": retry,
        "total_model_calls": total_model_calls,
        "retry_history": attempts,
    }


# ======================================================================
# Result writing
# ======================================================================

def _strip_raw_critic(verif: dict, cap: int = 2000) -> dict:
    """Trim raw_critic fields so JSONL records stay manageable."""
    v = dict(verif)
    for key in ("logic", "traceability"):
        if key in v and isinstance(v[key], dict):
            inner = dict(v[key])
            rc = inner.get("raw_critic", "")
            inner["raw_critic"] = rc[:cap] if isinstance(rc, str) else rc
            v[key] = inner
    return v


def write_results(
    records: list[dict],
    out_jsonl: Path,
    out_summary: Path,
    mode: str,
    backend: str,
    split_tag: str,
) -> None:
    """Write per-query JSONL and aggregate summary JSON."""
    # --- JSONL ---
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    # --- Summary ---
    total = len(records)
    correct_answer = sum(1 for r in records if r.get("answer_correct"))
    accuracy = correct_answer / total if total else 0.0

    cat_counts: dict[str, int] = {
        "correct": 0,
        "f_A_incorrect": 0,
        "A_neq_C": 0,
        "both_errors": 0,
    }
    cat_ids: dict[str, list[str]] = {k: [] for k in cat_counts}

    for r in records:
        cat = (r.get("verification") or {}).get("category", "unknown")
        if cat in cat_counts:
            cat_counts[cat] += 1
            cat_ids[cat].append(r.get("id", "?"))

    # Cross-tabulation: answer correctness vs reasoning category
    cross: dict[str, dict[str, int]] = {}
    for r in records:
        cat = (r.get("verification") or {}).get("category", "unknown")
        ans = "answer_correct" if r.get("answer_correct") else "answer_wrong"
        cross.setdefault(cat, {"answer_correct": 0, "answer_wrong": 0})
        cross[cat][ans] += 1

    summary: dict = {
        "mode": mode,
        "backend": backend,
        "split": split_tag,
        "total": total,
        "answer_accuracy": {
            "correct": correct_answer,
            "total": total,
            "rate": round(accuracy, 4),
        },
        "reasoning_categories": cat_counts,
        "category_ids": cat_ids,
        "cross_tabulation": cross,
    }

    if mode == "react_tool_verify":
        retries = [r.get("retries", 0) for r in records]
        summary["retry_stats"] = {
            "total_retries": sum(retries),
            "max_retries_used": max(retries) if retries else 0,
            "avg_retries": round(sum(retries) / len(retries), 2) if retries else 0,
            "fixed_by_retry": sum(
                1 for r in records
                if r.get("retries", 0) > 0 and r.get("answer_correct")
            ),
        }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # --- Console summary ---
    print()
    print("=" * 60)
    print(f"  Mode: {mode}  |  Backend: {backend}  |  Split: {split_tag}")
    print(f"  Answer accuracy: {correct_answer}/{total} = {accuracy:.4f}")
    print()
    print("  Reasoning categories:")
    for cat, cnt in cat_counts.items():
        pct = cnt / total * 100 if total else 0
        print(f"    {cat:20s}: {cnt:4d}  ({pct:5.1f}%)")
    print()
    print("  Cross-tabulation (category × answer correctness):")
    for cat, vals in cross.items():
        print(f"    {cat:20s}: correct={vals['answer_correct']:3d}  wrong={vals['answer_wrong']:3d}")
    if mode == "react_tool_verify":
        rs = summary["retry_stats"]
        print()
        print(f"  Retries: total={rs['total_retries']}  avg={rs['avg_retries']}  "
              f"fixed_by_retry={rs['fixed_by_retry']}")
    print("=" * 60)
    print(f"  JSONL:   {out_jsonl}")
    print(f"  Summary: {out_summary}")
    print()


# ======================================================================
# Main
# ======================================================================

def main() -> int:
    repo = _TASC_ROOT.parent

    ap = argparse.ArgumentParser(
        description="TASC experiment: CoT / ReAct+tool / ReAct+tool+verification-loop",
    )
    ap.add_argument(
        "--mode",
        choices=("cot", "react_tool", "react_tool_verify"),
        required=True,
        help="Experiment mode",
    )
    ap.add_argument(
        "--backend",
        choices=("ollama", "gemini"),
        default="ollama",
        help="LLM backend (default: ollama)",
    )
    ap.add_argument(
        "--gemini-model",
        default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Gemini model id (default: gemini-2.5-flash or GEMINI_MODEL env)",
    )
    ap.add_argument(
        "--split",
        choices=("train", "dev", "test"),
        default="dev",
        help="FinQA JSON split (default: dev). Ignored if --data is set.",
    )
    ap.add_argument(
        "--data", type=Path, default=None,
        help="Path to FinQA JSON (overrides --split)",
    )
    ap.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Max examples to run (default: all after --offset)",
    )
    ap.add_argument("--offset", type=int, default=0, help="Skip first N examples")
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (default: tasc/results)",
    )
    ap.add_argument(
        "--timeout", type=int, default=600,
        help="HTTP timeout seconds per LLM call",
    )
    ap.add_argument(
        "--max-retries", type=int, default=3,
        help="Max verification-loop retries (react_tool_verify only, default: 3)",
    )
    ap.add_argument("--debug", action="store_true", help="Print tool call details")
    args = ap.parse_args()

    # --- Data ---
    data_path = (
        args.data
        if args.data is not None
        else finqa_format.default_finqa_json_path(repo, args.split)
    )
    split_tag = args.split if args.data is None else data_path.stem
    if not data_path.is_file():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        return 1

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    end = args.offset + args.limit if args.limit is not None else len(data)
    subset = data[args.offset:end]

    # --- Output paths ---
    results_dir = args.out_dir or (_TASC_ROOT / "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"{args.mode}_{args.backend}_{split_tag}_{ts}"
    out_jsonl = results_dir / f"{base_name}.jsonl"
    out_summary = results_dir / f"{base_name}_summary.json"

    # --- Banner ---
    print(f"Mode:    {args.mode}")
    print(f"Data:    {data_path}")
    print(f"Subset:  [{args.offset}:{args.offset + len(subset)}] → {len(subset)} items")
    print(f"Backend: {args.backend}" + (f"  model={args.gemini_model}" if args.backend == "gemini" else ""))
    if args.mode == "react_tool_verify":
        print(f"Max retries: {args.max_retries}")
    print(f"Output:  {out_jsonl}")
    print()

    # --- Critic function (same backend, used for verification) ---
    critic_fn = _make_critic_fn(args.backend, args.gemini_model, args.timeout)

    # --- Main loop ---
    records: list[dict] = []
    correct_count = 0
    run_count = 0

    for i, ex in enumerate(subset):
        eid = ex.get("id", f"row_{args.offset + i}")
        qa = ex.get("qa") or {}
        gold = qa.get("answer")
        if gold is None:
            print(f"[{i + 1}/{len(subset)}] skip {eid} (no qa.answer)")
            continue
        gold = str(gold).strip()
        if not gold:
            print(f"[{i + 1}/{len(subset)}] skip {eid} (empty qa.answer)")
            continue

        run_count += 1
        print(f"[{i + 1}/{len(subset)}] {eid} ...", flush=True)

        try:
            if args.mode == "cot":
                result = run_cot_pipeline(
                    ex, gold,
                    backend=args.backend, gemini_model=args.gemini_model,
                    timeout=args.timeout, critic_fn=critic_fn,
                )
            elif args.mode == "react_tool":
                result = run_react_pipeline(
                    ex, gold,
                    backend=args.backend, gemini_model=args.gemini_model,
                    timeout=args.timeout, critic_fn=critic_fn, debug=args.debug,
                )
            elif args.mode == "react_tool_verify":
                result = run_react_verify_pipeline(
                    ex, gold,
                    backend=args.backend, gemini_model=args.gemini_model,
                    timeout=args.timeout, critic_fn=critic_fn,
                    max_retries=args.max_retries, debug=args.debug,
                )
            else:
                raise ValueError(f"Unknown mode {args.mode!r}")
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            result = {
                "raw": "",
                "reasoning": "",
                "predicted": None,
                "answer_correct": False,
                "verification": {
                    "logic": {"check": "logic_fA", "passed": False, "reason": f"generation error: {e}", "raw_critic": ""},
                    "traceability": {"check": "traceability_A_eq_C", "passed": False, "reason": f"generation error: {e}", "raw_critic": ""},
                    "category": "both_errors",
                },
                "model_calls": 0,
                "tool_used": False,
                "observation": None,
            }

        ok = result.get("answer_correct", False)
        if ok:
            correct_count += 1

        verif = result.get("verification") or {}
        cat = verif.get("category", "?")

        status = "OK" if ok else "FAIL"
        print(f"  {status}  gold={gold!r}  pred={result.get('predicted')!r}  category={cat}")
        if args.mode == "react_tool_verify" and result.get("retries", 0) > 0:
            print(f"    retries={result['retries']}")
        print()

        rec = {
            "id": eid,
            "mode": args.mode,
            "split": split_tag,
            "backend": args.backend,
            "gold_qa_answer": gold,
            "gold_exe_ans": qa.get("exe_ans"),
            "predicted": result.get("predicted"),
            "answer_correct": ok,
            "reasoning": (result.get("reasoning") or "")[:16000],
            "verification": _strip_raw_critic(verif),
            "model_calls": result.get("model_calls", 0),
            "tool_used": result.get("tool_used", False),
            "observation": result.get("observation"),
        }
        if args.backend == "gemini":
            rec["gemini_model"] = args.gemini_model
        if args.mode == "react_tool_verify":
            rec["retries"] = result.get("retries", 0)
            rec["total_model_calls"] = result.get("total_model_calls", 0)
            history = result.get("retry_history") or []
            for h in history:
                if "verification" in h:
                    h["verification"] = _strip_raw_critic(h["verification"])
                h["reasoning"] = (h.get("reasoning") or "")[:8000]
            rec["retry_history"] = history

        # Append to JSONL incrementally so partial runs are not lost
        with open(out_jsonl, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

        records.append(rec)

    # --- Final summary ---
    if run_count == 0:
        print("No examples processed.")
        return 0

    write_results(records, out_jsonl, out_summary, args.mode, args.backend, split_tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
