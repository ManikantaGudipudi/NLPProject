#!/usr/bin/env python3
"""
Numeric baseline on FinQA: LLM answers vs gold ``qa.answer`` (same scoring for all backends).

Usage (from repo root NLPProject):
  # Ollama (local)
  python tasc/scripts/run_numeric_baseline.py --limit 5
  python tasc/scripts/run_numeric_baseline.py --split train --limit 50
  OLLAMA_MODEL=llama3.2:3b python tasc/scripts/run_numeric_baseline.py --limit 3

  # Google Gemini (needs GEMINI_API_KEY in env — never commit the key)
  export GEMINI_API_KEY=...
  python tasc/scripts/run_numeric_baseline.py --backend gemini --gemini-model gemini-2.5-flash --limit 3

  # Chain-of-thought: reasoning logged (still scores on last FINAL: line)
  python tasc/scripts/run_numeric_baseline.py --cot --limit 2

  # Full dev split (omit --limit)
  # python tasc/scripts/run_numeric_baseline.py

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

# tasc/ on path
_TASC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TASC_ROOT))

import finqa_format  # noqa: E402
import gemini_client  # noqa: E402
import numeric_eval  # noqa: E402
import ollama_client  # noqa: E402

SYSTEM_PROMPT = """You are a precise financial question-answering assistant.

You must answer using only the passage and table provided in the user message.
Do not use outside knowledge.
Do not guess.
If the required information is not explicitly available, infer only from values directly present in the passage/table.

Your job is to compute or decide the answer and return exactly one final line in this format:
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
- Do not output reasoning
- Do not output any text before or after the FINAL line

Examples (format only):
Question: what is 15% of 200?
FINAL: 30

Question: what fraction of revenue is from services if services are 637 and total is 1000?
FINAL: 0.637

Question: did revenue exceed 500 in 2019?
FINAL: yes
"""

USER_SUFFIX = """

Answer the question using only the passage/table above.
Return exactly one line:
FINAL: <value>
"""

# CoT: structured reasoning; scoring still uses the last ``FINAL:`` line.
SYSTEM_PROMPT_COT = """You are a precise financial question-answering assistant.

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

USER_SUFFIX_COT = """

Answer the question using only the passage/table above.

Show concise step-by-step reasoning in this order:
1. RELEVANT VALUES
2. OPERATION
3. COMPUTATION / DECISION
4. ANSWER CHECK

Then end with exactly one final line:
FINAL: <value>
"""


def _generate(
    backend: str,
    user_msg: str,
    *,
    system_prompt: str,
    gemini_model: str | None,
    timeout: int,
    cot: bool,
) -> str:
    ollama_num_predict = int(
        os.environ.get(
            "OLLAMA_NUM_PREDICT",
            str(2048 if cot else 256),
        )
    )
    if backend == "ollama":
        return ollama_client.ollama_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            timeout=timeout,
            options={
                "temperature": float(os.environ.get("OLLAMA_TEMPERATURE", "0.1")),
                "num_predict": ollama_num_predict,
            },
        )
    if backend == "gemini":
        return gemini_client.generate_content(
            system_prompt,
            user_msg,
            model=gemini_model,
            timeout=timeout,
            max_output_tokens=8192 if cot else 4096,
        )
    raise ValueError(f"Unknown --backend {backend!r}")


def main() -> int:
    repo = _TASC_ROOT.parent

    ap = argparse.ArgumentParser(
        description="FinQA numeric baseline (Ollama or Gemini)"
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
        help="Append JSONL log (default: tasc/results/numeric_baseline_<backend>_<split>_<ts>.jsonl)",
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
        help="Chain-of-thought: structured reasoning, then log ``reasoning`` before FINAL line (still scores on FINAL:)",
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
            / f"numeric_baseline_{args.backend}_{split_tag}{suffix}_{ts}.jsonl"
        )

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    end = args.offset + args.limit if args.limit is not None else len(data)
    subset = data[args.offset : end]
    print(f"Data: {data_path}")
    print(f"Loaded {len(data)} examples; running [{args.offset}:{args.offset + len(subset)}] -> {len(subset)} items")
    print(f"Backend: {args.backend}" + (f"  model={args.gemini_model}" if args.backend == "gemini" else ""))
    print(f"CoT: {args.cot}")
    print(f"Log: {out_path}")
    print()

    system_prompt = SYSTEM_PROMPT_COT if args.cot else SYSTEM_PROMPT
    user_suffix = USER_SUFFIX_COT if args.cot else USER_SUFFIX

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
        context = finqa_format.format_finqa_context(ex)
        user_msg = context + user_suffix

        print(f"[{i + 1}/{len(subset)}] {eid} ...", flush=True)
        try:
            raw = _generate(
                args.backend,
                user_msg,
                system_prompt=system_prompt,
                gemini_model=args.gemini_model,
                timeout=args.timeout,
                cot=args.cot,
            )
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            raw = ""
            ok = False
            parsed = None
        else:
            parsed = numeric_eval.parse_predicted_answer(raw)
            ok = numeric_eval.answer_matches_qa_answer(gold, raw)
            if ok:
                correct += 1

        raw_cap = 32000 if args.cot else 8000
        rec = {
            "id": eid,
            "split": split_tag,
            "backend": args.backend,
            "cot": args.cot,
            "gold_qa_answer": gold,
            "gold_exe_ans": gold_exe,
            "parsed": parsed,
            "correct": ok,
            "raw_reply": raw[:raw_cap],
        }
        if args.cot:
            rec["reasoning"] = numeric_eval.extract_reasoning_before_final(raw)
        if args.backend == "gemini":
            rec["gemini_model"] = args.gemini_model
        with open(out_path, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        status = "OK" if ok else "FAIL"
        print(f"  {status}  qa.answer={gold!r}  parsed={parsed!r}")
        if args.cot and rec.get("reasoning"):
            r = rec["reasoning"]
            preview = r
            print(f"  reasoning preview: {preview}")
        print()

    acc = correct / run if run else 0.0
    print(f"Answer accuracy vs qa.answer: {correct}/{run} = {acc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
