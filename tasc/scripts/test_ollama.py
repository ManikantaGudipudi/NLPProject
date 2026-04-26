#!/usr/bin/env python3
"""
Smoke test: verify Ollama is running and a Llama model responds.

Usage:
  python test_ollama.py
  OLLAMA_MODEL=llama3.2:3b python test_ollama.py
  OLLAMA_HOST=http://127.0.0.1:11434 python test_ollama.py

Requires: Ollama installed and running (`ollama serve`), model pulled once.
"""

from __future__ import annotations

import os
import sys
import urllib.error
from pathlib import Path

# tasc/ on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ollama_client  # noqa: E402


def main() -> int:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

    print(f"Ollama host: {host}")
    print(f"Model:       {model}")
    print()

    # 1) Health: list local models
    try:
        tags = ollama_client.get_json(f"{host}/api/tags")
    except urllib.error.URLError as e:
        print("ERROR: Cannot reach Ollama. Is `ollama serve` running?", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        return 1

    names = [m.get("name", "") for m in tags.get("models", [])]
    print("Local models:", ", ".join(names) if names else "(none)")
    if not any(
        n == model or n.startswith(model + ":") or model.startswith(n.split(":")[0])
        for n in names
    ):
        print()
        print(
            f"WARNING: '{model}' not found in the list above.",
            "Pull it once, e.g.:",
            sep="\n  ",
        )
        print(f"  ollama pull {model}")
        print()

    # 2) Chat completion
    prompt = "Reply with exactly one short sentence confirming you are working."
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    print("Sending test prompt...")
    try:
        out = ollama_client.post_json(f"{host}/api/chat", payload, timeout=300)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print("ERROR: /api/chat failed:", e.code, body[:500], file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print("ERROR:", e, file=sys.stderr)
        return 1

    msg = out.get("message", {})
    content = msg.get("content", "").strip()
    print()
    print("--- Model reply ---")
    print(content or "(empty)")
    print("--- End ---")
    print()

    if not content:
        print("WARNING: Empty reply; check model name and Ollama logs.")
        return 2

    print("OK: Ollama + model responded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
