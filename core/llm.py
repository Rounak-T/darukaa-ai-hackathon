"""Provider-agnostic LLM wrapper with automatic fallback."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


class LLMError(RuntimeError):
    pass


def _gemini(messages: List[Dict[str, str]], system: Optional[str], temperature: float) -> str:
    if not GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in messages]
    payload: Dict[str, Any] = {"contents": contents, "generationConfig": {"temperature": temperature, "maxOutputTokens": 2048}}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    r = requests.post(url, headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}, json=payload, timeout=90)
    if r.status_code != 200:
        raise LLMError(f"Gemini {r.status_code}: {r.text[:300]}")
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Unexpected Gemini payload: {json.dumps(data)[:300]}") from exc


def _huggingface(messages: List[Dict[str, str]], system: Optional[str], temperature: float) -> str:
    if not HF_API_KEY:
        raise LLMError("HF_API_KEY not set")
    msgs = ([{"role": "system", "content": system}] if system else []) + messages
    r = requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers={"Authorization": f"Bearer {HF_API_KEY}"},
        json={"model": HF_MODEL, "messages": msgs, "temperature": temperature, "max_tokens": 2048},
        timeout=120,
    )
    if r.status_code != 200:
        raise LLMError(f"HuggingFace {r.status_code}: {r.text[:300]}")
    return r.json()["choices"][0]["message"]["content"]


def complete(prompt: str, system: Optional[str] = None, temperature: float = 0.3) -> str:
    messages = [{"role": "user", "content": prompt}]
    order = ["gemini", "huggingface"] if LLM_PROVIDER == "gemini" else ["huggingface", "gemini"]
    last_error: Optional[Exception] = None
    for p in order:
        try:
            return _gemini(messages, system, temperature) if p == "gemini" else _huggingface(messages, system, temperature)
        except Exception as exc:
            last_error = exc
            continue
    raise LLMError(f"All providers failed. Last error: {last_error}")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def complete_json(prompt: str, system: Optional[str] = None, temperature: float = 0.2) -> Any:
    """Ask for JSON, parse it defensively - free-tier models often wrap JSON in code fences."""
    raw = complete(prompt + "\n\nRespond with valid JSON only. No prose, no markdown fences.", system=system, temperature=temperature)
    text = raw.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for opener, closer in (("{", "}"), ("[", "]")):
            start, end = text.find(opener), text.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    continue
    raise LLMError(f"Could not parse JSON from model output: {raw[:300]}")