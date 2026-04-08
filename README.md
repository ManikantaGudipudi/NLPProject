# NLPProject

Course / experiments repo. **Tracked code:** mainly **`tasc/`** (FinQA LLM baselines).

**`FinQA-main/`** and **`TAT-QA-master/`** are **not** in Git (too large for GitHub). Add them locally at the **repository root** next to `tasc/`.

---

## 1. Add datasets / upstream trees locally

From the **NLPProject** root (same folder as this `README.md`):

### FinQA (required for `tasc` baselines)

```bash
git clone https://github.com/czyssrs/FinQA.git FinQA-main
```

You should have **`FinQA-main/dataset/train.json`**, **`dev.json`**, **`test.json`** after clone. Baselines read **`FinQA-main/dataset/<split>.json`**.

Avoid committing Hugging Face caches inside `FinQA-main` (e.g. remove **`FinQA-main/.cache/`** if you do not need it, or keep it only on disk).

### TAT-QA

Clone or unpack the TAT-QA release your course uses into:

```text
TAT-QA-master/
```

at the same root (only if you run TAT-QA-related work).

---

## 2. Run baselines (`tasc/`)

Python 3. Install Ollama and a model locally, **or** set **`GEMINI_API_KEY`** for Gemini.

```bash
# smoke test (Ollama)
python3 tasc/scripts/test_ollama.py

# numeric: single line FINAL: … vs qa.answer
python3 tasc/scripts/run_numeric_baseline.py --limit 10
python3 tasc/scripts/run_numeric_baseline.py --split dev --limit 50
python3 tasc/scripts/run_numeric_baseline.py --backend gemini --gemini-model gemini-2.5-flash --limit 5

# ReAct: calculator tool, ≤2 LLM calls, Final Answer: …
python3 tasc/scripts/run_react_baseline.py --limit 10
python3 tasc/scripts/run_react_baseline.py --backend gemini --gemini-model gemini-2.5-flash --limit 5
```

Common CLI options:

- `--split train|dev|test` (default `dev`)
- `--data PATH` — use a different JSON file
- `--limit N` — omit to run the full split
- `--cot`, `--out PATH`
- Logs: `tasc/results/*.jsonl`

Secrets: use `tasc/.env` from `tasc/.env.example`; never commit API keys.