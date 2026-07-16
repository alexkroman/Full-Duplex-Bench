#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llm_judge.py

Shared LLM judge (gpt-4o) helpers used by evaluate_tool_calls.py and
evaluate_pass_rate.py: semantic argument matching, spoken-response judging,
and the exact-match fallback when no LLM is available.
"""

import json
from typing import Tuple

_openai_client = None


def _strip_json_fences(text: str) -> str:
    """Remove markdown ```json ... ``` code fences that gpt-4o sometimes adds."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # drop opening fence
        lines = lines[1:]
        # drop closing fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _get_openai_client():
    """Lazy-init OpenAI client."""
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            _openai_client = OpenAI()
        except Exception as e:
            print(f"⚠️  OpenAI client not available: {e}")
            _openai_client = None
    return _openai_client


def llm_judge_argument(expected_args: dict, actual_args: dict, function_name: str) -> Tuple[bool, str]:
    """
    Use gpt-4o to judge if actual function arguments semantically match expected.
    Returns (is_correct: bool, explanation: str).
    """
    client = _get_openai_client()
    if client is None:
        return exact_match_args(expected_args, actual_args)

    prompt = f"""You are evaluating whether an AI voice agent called a function with correct arguments.

Function: {function_name}
Expected arguments: {json.dumps(expected_args)}
Actual arguments: {json.dumps(actual_args)}

Rules:
1. Arguments that start with "$" (like "$RESULT_0.flights[0].flight_id") are dynamic references —
   the actual value should be any real value that could plausibly come from a previous API call.
2. Minor formatting differences are fine: "August 20" == "2026-08-20", "New York" == "new york".
3. "Las Vegas" == "Vegas" — abbreviations and common aliases are acceptable.
4. Numeric tolerance: ±5% is acceptable.
5. doc_type: "driver_license" == "driver license" (underscore vs space).

Respond with ONLY a JSON object:
{{"correct": true/false, "explanation": "brief reason"}}"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        raw = _strip_json_fences(resp.choices[0].message.content)
        result = json.loads(raw)
        return result["correct"], result["explanation"]
    except Exception:
        return exact_match_args(expected_args, actual_args)


def llm_judge_response(expected_intent: str, actual_transcript: str) -> Tuple[float, str]:
    """
    Use gpt-4o to judge if the agent's spoken response matches the expected intent.
    Returns (score: float 0.0 or 1.0, explanation: str).
    """
    if not actual_transcript or not actual_transcript.strip():
        return 0.0, "No transcript available."

    client = _get_openai_client()
    if client is None:
        return 0.0, "OpenAI client unavailable."

    prompt = f"""You are evaluating whether an AI voice agent successfully completed the user's requested task.

Expected Task/Action: "{expected_intent}"
Actual Agent Spoken Response: "{actual_transcript}"

Evaluation criteria:
1. Did the agent perform the CORRECT actions (right tools, right parameters)?
2. Did the response indicate the task was completed or is being handled?
3. It is FINE if the agent provides MORE detail than expected (e.g., giving specific results, prices, confirmation numbers). Providing additional helpful information is NOT a penalty.
4. It is INCORRECT if the agent says it cannot perform the action, lacks tools, or refuses.
5. It is INCORRECT if the agent performs the WRONG action (e.g., wrong destination, wrong document type).
6. Partial delivery of multi-step tasks (e.g., completes 2 of 3 required steps) should be scored 0.

Respond with ONLY a JSON object:
{{"correct": true/false, "explanation": "brief reason"}}"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        raw = _strip_json_fences(resp.choices[0].message.content)
        result = json.loads(raw)
        is_correct = result.get("correct", False)
        return (1.0 if is_correct else 0.0), result.get("explanation", "")
    except Exception as e:
        return 0.0, f"LLM parsing error: {str(e)}"


def exact_match_args(expected: dict, actual: dict) -> Tuple[bool, str]:
    """Fallback exact-match for arguments."""
    def normalize(v):
        if isinstance(v, str):
            return v.lower().strip().replace("_", " ")
        return v

    for key, exp_val in expected.items():
        if key not in actual:
            return False, f"Missing argument: {key}"
        if isinstance(exp_val, str) and exp_val.startswith("$"):
            continue  # Dynamic reference, skip exact check
        if normalize(exp_val) != normalize(actual.get(key)):
            return False, f"Mismatch '{key}': expected={exp_val}, got={actual.get(key)}"
    return True, "All arguments match"
