#!/usr/bin/env python3
"""
Gemini API client helpers used by eval_single_item.py:
retrying generateContent calls and sanitizing JSON out of model responses.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any, Dict, Optional

import requests


# ---------------------------
# Gemini API call with retries
# ---------------------------

def generate_with_gemini(
    api_key: str,
    model: str,
    api_version: str,
    prompt_text: str,
    max_output_tokens: int = 20000,
    temperature: float = 0.2,
    max_retries: int = 5,
) -> Dict[str, Any]:
    base = "https://generativelanguage.googleapis.com"
    # support v1beta for preview models
    url = f"{base}/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    payload: Dict[str, Any] = {
        "contents": [
            {
                "parts": [{"text": prompt_text}],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "temperature": temperature,
        },
    }

    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
        except requests.RequestException as e:
            if attempt >= max_retries:
                raise RuntimeError(f"HTTP request failed after {attempt} attempts: {e}")
            sleep_s = min(60, 2 ** attempt + random.random())
            time.sleep(sleep_s)
            continue

        if resp.status_code == 200:
            try:
                return resp.json()
            except json.JSONDecodeError as e:
                if attempt >= max_retries:
                    raise RuntimeError(f"Invalid JSON response after {attempt} attempts: {e}")
                sleep_s = min(60, 2 ** attempt + random.random())
                time.sleep(sleep_s)
                continue

        if resp.status_code in (429, 500, 502, 503, 504):
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Gemini API error {resp.status_code} after {attempt} attempts: {resp.text}"
                )
            retry_after = resp.headers.get("Retry-After")
            try:
                sleep_s = float(retry_after) if retry_after else None
            except ValueError:
                sleep_s = None
            if sleep_s is None:
                sleep_s = min(60, 2 ** attempt + random.random())
            time.sleep(sleep_s)
            continue

        # Non-retryable
        raise RuntimeError(
            f"Gemini API returned {resp.status_code}: {resp.text}"
        )


def extract_text_from_response(resp: Dict[str, Any]) -> str:
    try:
        cands = resp.get("candidates") or []
        if not cands:
            return ""
        content = cands[0].get("content") or {}
        parts = content.get("parts") or []
        if not parts:
            return ""
        text = parts[0].get("text") or ""
        return text
    except Exception:
        return ""


# ---------------------------
# JSON sanitization helpers
# ---------------------------

def strip_markdown_fences(text: str) -> str:
    """Remove surrounding markdown code fences like ```json ... ``` if present.

    Keeps inner content intact. Returns original text if no fences detected.
    """
    t = text.strip()
    if t.startswith("```"):
        # Drop the first fence line
        first_newline = t.find("\n")
        if first_newline != -1:
            t = t[first_newline + 1 :]
        # Drop trailing fence if present
        if t.endswith("```"):
            t = t[: -3]
    return t.strip()


def extract_first_balanced_json(text: str) -> Optional[str]:
    """Extract the first top-level balanced JSON object from arbitrary text.

    Scans for the first '{' and returns the shortest string that balances braces.
    Returns None if no balanced object is found.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    return candidate
    return None


def try_parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse: strip fences, extract balanced JSON, fall back to raw loads.

    Returns a dict if successful, otherwise None.
    """
    if not text:
        return None
    stripped = strip_markdown_fences(text)
    candidate = extract_first_balanced_json(stripped)
    if candidate is None:
        candidate = stripped
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None
