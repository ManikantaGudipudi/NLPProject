#!/usr/bin/env python3
"""
Minimal ReAct baseline on FinQA: at most 2 model calls, at most one calculator use.

Usage (from repo root NLPProject):
  python tasc/scripts/run_react_baseline.py --limit 5
  python tasc/scripts/run_react_baseline.py --split test --limit 20
  OLLAMA_MODEL=deepseek-r1:32b python tasc/scripts/run_react_baseline.py --limit 3

  export GEMINI_API_KEY=...
  python tasc/scripts/run_react_baseline.py --backend gemini --gemini-model gemini-2.5-flash --limit 3
  python tasc/scripts/run_react_baseline.py --limit 2 --debug   # print calculator tool args when used

  # Full dev split: omit --limit
  # python tasc/scripts/run_react_baseline.py

Default data: ``FinQA-main/dataset/<split>.json`` (``--split dev|train|test``).
Omit ``--limit`` to run the full split (after ``--offset``).
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


def _call_llm(
    backend: str,
    messages: list[dict[str, str]],
    *,
    system_prompt: str,
    gemini_model: str | None,
    timeout: int,
    cot: bool,
) -> str:
    ollama_num_predict = int(
        os.environ.get(
            "OLLAMA_NUM_PREDICT",
            str(2048 if cot else 1024),
        )
    )
    if backend == "ollama":
        return ollama_client.ollama_chat(
            [{"role": "system", "content": system_prompt}, *messages],
            timeout=timeout,
            options={
                "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", "0.1")),
                "num_predict": ollama_num_predict,
            },
        )
    if backend == "gemini":
        return gemini_client.generate_content_chat(
            messages,
            system_instruction=system_prompt,
            model=gemini_model,
            timeout=timeout,
            max_output_tokens=8192 if cot else 4096,
        )
    raise ValueError(f"Unknown --backend {backend!r}")


def _score_react(gold: str, full_text: str) -> bool:
    return numeric_eval.answer_matches_react_qa_answer(gold, full_text)


def run_react_one(
    example: dict,
    *,
    backend: str,
    system_prompt: str,
    user_suffix: str,
    gemini_model: str | None,
    timeout: int,
    cot: bool,
    debug: bool = False,
) -> dict:
    """
    Return a record dict: raw turns, parsed, observation, correct, etc.
    """
    context = finqa_format.format_finqa_context(example)
    user1 = context + user_suffix

    messages1: list[dict[str, str]] = [{"role": "user", "content": user1}]
    raw1 = _call_llm(
        backend,
        messages1,
        system_prompt=system_prompt,
        gemini_model=gemini_model,
        timeout=timeout,
        cot=cot,
    )
    p1 = react_parse.parse_react_block(raw1)

    final_text = raw1
    raw2 = ""
    observation: str | None = None
    calls = 1

    use_calc = react_parse.wants_calculator(p1)

    if use_calc:
        expr = (p1.get("action_input") or "").strip()
        if debug:
            print(
                f"  DEBUG tool_call: action={p1.get('action')!r} action_input={expr!r}",
                flush=True,
            )
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
        raw2 = _call_llm(
            backend,
            messages2,
            system_prompt=system_prompt,
            gemini_model=gemini_model,
            timeout=timeout,
            cot=cot,
        )
        calls = 2
        final_text = raw2

    return {
        "raw_turn1": raw1,
        "parsed_turn1": p1,
        "observation": observation,
        "raw_turn2": raw2 if raw2 else None,
        "final_text": final_text,
        "model_calls": calls,
    }


def main() -> int:
    repo = _TASC_ROOT.parent

    ap = argparse.ArgumentParser(
        description="FinQA minimal ReAct baseline (calculator only; max 2 LLM calls)"
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
        "--data",
        type=Path,
        default=None,
        help="Path to FinQA JSON (overrides --split)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Max examples to run (default: all after --offset)",
    )
    ap.add_argument("--offset", type=int, default=0, help="Skip first N examples")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Append JSONL log (default: tasc/results/react_baseline_<backend>_<split>_<ts>.jsonl)",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="HTTP timeout seconds (Ollama or Gemini)",
    )
    ap.add_argument(
        "--cot",
        action="store_true",
        help="Larger token budget (OLLAMA_NUM_PREDICT / Gemini max_output_tokens)",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Print parsed tool call arguments (calculator action_input) when used",
    )
    args = ap.parse_args()

    data_path = args.data if args.data is not None else finqa_format.default_finqa_json_path(
        repo, args.split
    )
    split_tag = args.split if args.data is None else data_path.stem
    if not data_path.is_file():
        print(f"ERROR: data file not found: {data_path}", file=sys.stderr)
        return 1

    out_path = args.out
    if out_path is None:
        results_dir = _TASC_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = "_cot" if args.cot else ""
        out_path = (
            results_dir
            / f"react_baseline_{args.backend}_{split_tag}{suffix}_{ts}.jsonl"
        )

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    end = args.offset + args.limit if args.limit is not None else len(data)
    subset = data[args.offset : end]
    print(f"Data: {data_path}")
    print(f"Loaded {len(data)} examples; running [{args.offset}:{args.offset + len(subset)}] -> {len(subset)} items")
    print(f"Backend: {args.backend}" + (f"  model={args.gemini_model}" if args.backend == "gemini" else ""))
    print(f"CoT (longer budget): {args.cot}")
    print(f"Log: {out_path}")
    print()

    system_prompt = react_prompts.REACT_SYSTEM_PROMPT
    user_suffix = react_prompts.REACT_USER_SUFFIX

    correct = 0
    run = 0
    for i, ex in enumerate(subset):
        eid = ex.get("id", f"row_{args.offset + i}")
        qa = ex.get("qa") or {}
        gold = qa.get("answer")
        if gold is None:
            print(f"[{i + 1}/{len(subset)}] skip {eid} (no qa.answer)")
            continue
        if not isinstance(gold, str):
            gold = str(gold)
        if not gold.strip():
            print(f"[{i + 1}/{len(subset)}] skip {eid} (empty qa.answer)")
            continue

        run += 1
        gold_exe = qa.get("exe_ans")
        print(f"[{i + 1}/{len(subset)}] {eid} ...", flush=True)
        try:
            rec = run_react_one(
                ex,
                backend=args.backend,
                system_prompt=system_prompt,
                user_suffix=user_suffix,
                gemini_model=args.gemini_model,
                timeout=args.timeout,
                cot=args.cot,
                debug=args.debug,
            )
            final_text = rec["final_text"]
            ok = _score_react(gold, final_text)
            parsed_display = numeric_eval.extract_react_final_answer(final_text)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            rec = {
                "raw_turn1": "",
                "parsed_turn1": {},
                "observation": None,
                "raw_turn2": None,
                "final_text": "",
                "model_calls": 0,
            }
            ok = False
            parsed_display = None

        if ok:
            correct += 1

        raw_cap = 32000 if args.cot else 16000
        rec_out = {
            "id": eid,
            "split": split_tag,
            "backend": args.backend,
            "cot": args.cot,
            "gold_qa_answer": gold,
            "gold_exe_ans": gold_exe,
            "parsed": parsed_display,
            "parsed_turn1": rec.get("parsed_turn1"),
            "correct": ok,
            "model_calls": rec.get("model_calls"),
            "observation": rec.get("observation"),
            "raw_turn1": (rec.get("raw_turn1") or "")[:raw_cap],
            "raw_turn2": (rec.get("raw_turn2") or "")[:raw_cap] if rec.get("raw_turn2") else None,
        }
        if args.backend == "gemini":
            rec_out["gemini_model"] = args.gemini_model
        with open(out_path, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

        status = "OK" if ok else "FAIL"
        print(f"  {status}  qa.answer={gold!r}  parsed={parsed_display!r}  calls={rec.get('model_calls')}")
        print()

    acc = correct / run if run else 0.0
    print(f"Answer accuracy vs qa.answer: {correct}/{run} = {acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
