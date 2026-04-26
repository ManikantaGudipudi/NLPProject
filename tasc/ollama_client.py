"""Minimal Ollama HTTP client (stdlib only)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_chat(
    messages: list[dict],
    *,
    model: str | None = None,
    host: str | None = None,
    timeout: int = 600,
    options: dict | None = None,
) -> str:
    """
    POST /api/chat (non-streaming). Returns assistant message content.

    ``options`` is passed to Ollama (e.g. {"temperature": 0.1, "num_predict": 256}).
    """
    base = (host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
    model = model or os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    if options:
        payload["options"] = options
    try:
        out = post_json(f"{base}/api/chat", payload, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama /api/chat HTTP {e.code}: {body[:800]}") from e
    msg = out.get("message") or {}
    return (msg.get("content") or "").strip()


def list_local_models(host: str | None = None) -> list[str]:
    base = (host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
    tags = get_json(f"{base}/api/tags")
    return [m.get("name", "") for m in tags.get("models", [])]
