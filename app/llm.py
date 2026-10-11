"""Wrapper around the local Ollama server.

Everything goes through here so there is one place to change the model, and
one place that knows how to dig a JSON object out of a small model's reply -
qwen2.5:3b is good but it occasionally wraps JSON in prose or fences.
"""

import json
import re
from typing import Any

import ollama

from app.config import settings

_client: ollama.Client | None = None

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def client() -> ollama.Client:
    global _client
    if _client is None:
        _client = ollama.Client(host=settings.ollama_host)
    return _client


def chat(
    messages: list[dict],
    model: str | None = None,
    tools: list[dict] | None = None,
    temperature: float | None = None,
    json_mode: bool = False,
    max_tokens: int | None = None,
) -> dict:
    """Returns the raw ollama response dict.

    `max_tokens` maps to Ollama's num_predict. It is not optional in practice -
    an uncapped 3B model will sometimes just keep going and generate tens of
    thousands of tokens, which reads as a hang.
    """
    options = {
        "temperature": settings.temperature if temperature is None else temperature,
        "num_predict": max_tokens or settings.max_answer_tokens,
    }
    kwargs: dict[str, Any] = {
        "model": model or settings.chat_model,
        "messages": messages,
        "options": options,
    }
    if tools:
        kwargs["tools"] = tools
    if json_mode:
        kwargs["format"] = "json"
    return client().chat(**kwargs)


def message_of(response: dict) -> dict:
    return response.get("message") or {}


def content_of(response: dict) -> str:
    return (message_of(response).get("content") or "").strip()


def tool_calls_of(response: dict) -> list[dict]:
    return message_of(response).get("tool_calls") or []


def parse_json(text: str, default: Any = None) -> Any:
    """Best-effort JSON extraction from a model reply."""
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = _FENCE.search(text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return default


def generate_json(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
) -> Any:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = chat(
        messages,
        model=model,
        json_mode=True,
        max_tokens=max_tokens or settings.max_helper_tokens,
    )
    return parse_json(content_of(response))


def usage_of(response: dict) -> dict:
    keys = ("prompt_eval_count", "eval_count", "total_duration")
    return {k: response[k] for k in keys if response.get(k) is not None}


def merge_usage(*usages: dict | None) -> dict:
    """Add up usage dicts, as returned by `usage_of`, across several calls."""
    out: dict = {}
    for usage in usages:
        for key, value in (usage or {}).items():
            if value is not None:
                out[key] = out.get(key, 0) + value
    return out


def health() -> dict:
    """Used by /health and the ingest CLI to fail early with a clear message."""
    try:
        models = client().list().get("models", [])
        names = [m.get("model") or m.get("name") for m in models]
        return {
            "ok": True,
            "host": settings.ollama_host,
            "models": names,
            "chat_model_available": any(settings.chat_model in (n or "") for n in names),
        }
    except Exception as exc:  # server not running, wrong host, etc.
        return {"ok": False, "host": settings.ollama_host, "error": str(exc), "models": []}
