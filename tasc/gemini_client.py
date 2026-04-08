"""Google Gemini API (REST, stdlib only). Uses GEMINI_API_KEY from the environment."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request


def generate_content(
    system_instruction: str,
    user_text: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    timeout: int = 120,
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
) -> str:
    """
    Call generateContent (v1beta). Returns concatenated text from the first candidate.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "Missing API key: set environment variable GEMINI_API_KEY "
            "(never commit keys to git)."
        )
    model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    base = "https://generativelanguage.googleapis.com/v1beta"
    path = f"/models/{model}:generateContent"
    q = urllib.parse.urlencode({"key": key})
    url = f"{base}{path}?{q}"

    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_text}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Gemini HTTP {e.code}: {err_body[:1200]}"
        ) from e

    candidates = body.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {body!r}")

    parts = (candidates[0].get("content") or {}).get("parts") or []
    texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
    return "\n".join(texts).strip()


def generate_content_chat(
    messages: list[dict[str, str]],
    *,
    system_instruction: str,
    api_key: str | None = None,
    model: str | None = None,
    timeout: int = 120,
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
) -> str:
    """
    Multi-turn chat: ``messages`` are ``{"role": "user"|"assistant", "content": "..."}``.
    Gemini uses role ``model`` for assistant turns.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "Missing API key: set environment variable GEMINI_API_KEY "
            "(never commit keys to git)."
        )
    model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    base = "https://generativelanguage.googleapis.com/v1beta"
    path = f"/models/{model}:generateContent"
    q = urllib.parse.urlencode({"key": key})
    url = f"{base}{path}?{q}"

    contents: list[dict] = []
    for m in messages:
        role = m.get("role", "")
        text = (m.get("content") or "").strip()
        if not text:
            continue
        if role == "assistant":
            role = "model"
        if role not in ("user", "model"):
            raise ValueError(f"message role must be user or assistant, got {role!r}")
        contents.append({"role": role, "parts": [{"text": text}]})

    if not contents:
        raise ValueError("no messages")

    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Gemini HTTP {e.code}: {err_body[:1200]}"
        ) from e

    candidates = body.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {body!r}")

    parts = (candidates[0].get("content") or {}).get("parts") or []
    texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
    return "\n".join(texts).strip()
