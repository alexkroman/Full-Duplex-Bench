#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tool_call_metrics.py

Per-scenario metric functions for multi-step tool calling, used by
evaluate_tool_calls.py:

  1. tool_selection_acc  — F1 of recall & precision over expected vs actual tool calls
  2. argument_acc        — semantic correctness of arguments (LLM judge, gpt-4o)
  3. response_qual       — spoken response quality (LLM judge, gpt-4o)
  4. latency_metrics     — agent response latency + interruption rate
"""

from typing import List, Dict, Optional

from llm_judge import llm_judge_argument, llm_judge_response, exact_match_args


# ==============================================================================
# Metric 1: Tool Selection Accuracy (F1)
# ==============================================================================

def evaluate_tool_selection(expected_calls: List[dict], actual_calls: List[dict]) -> dict:
    """
    Score: F1 of recall and precision over tool names.

    Recall    = matched_expected / total_expected
    Precision = matched_expected / total_actual  (penalises extra calls)
    F1        = harmonic mean of recall and precision

    Handles duplicates via multiset matching.
    """
    expected_names = [c["function"] for c in expected_calls]
    actual_names   = [c["function"] for c in actual_calls]

    # Multiset intersection
    exp_remaining = list(expected_names)
    act_remaining = list(actual_names)
    matched = 0
    for fn in list(exp_remaining):
        if fn in act_remaining:
            matched += 1
            exp_remaining.remove(fn)
            act_remaining.remove(fn)

    total_expected = len(expected_names)
    total_actual   = len(actual_names)

    recall    = matched / total_expected if total_expected > 0 else 1.0
    precision = matched / total_actual   if total_actual   > 0 else 1.0

    if recall + precision > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0.0

    return {
        "score":            round(f1, 3),
        "recall":           round(recall, 3),
        "precision":        round(precision, 3),
        "matched":          matched,
        "total_expected":   total_expected,
        "total_actual":     total_actual,
        "unmatched_expected": exp_remaining,
        "unexpected_calls":   act_remaining,
    }


# ==============================================================================
# Metric 2: Argument Accuracy
# ==============================================================================

def evaluate_argument_accuracy(
    expected_calls: List[dict],
    actual_calls: List[dict],
    use_llm: bool = False,
) -> dict:
    """
    Score: For each expected call that was actually called, judge argument correctness.
    Calls that were NOT called at all score 0 but are noted separately.
    """
    if not expected_calls:
        return {"score": 1.0, "note": "No expected calls", "details": []}

    # Build map: function -> list of actual calls (FIFO for duplicates)
    actual_by_func: Dict[str, List[dict]] = {}
    for ac in actual_calls:
        actual_by_func.setdefault(ac["function"], []).append(ac)

    call_scores = []
    for ec in expected_calls:
        func = ec["function"]
        expected_args = ec.get("args", {})

        if func not in actual_by_func or not actual_by_func[func]:
            call_scores.append({
                "function": func,
                "score":    0.0,
                "reason":   "Function not called",
            })
            continue

        actual_call = actual_by_func[func].pop(0)
        actual_args = actual_call.get("args", {})

        if use_llm:
            is_ok, explanation = llm_judge_argument(expected_args, actual_args, func)
        else:
            is_ok, explanation = exact_match_args(expected_args, actual_args)

        call_scores.append({
            "function":     func,
            "score":        1.0 if is_ok else 0.0,
            "expected_args": expected_args,
            "actual_args":   actual_args,
            "explanation":   explanation,
        })

    avg = sum(c["score"] for c in call_scores) / len(call_scores) if call_scores else 0.0
    return {"score": round(avg, 3), "details": call_scores}


# ==============================================================================
# Metric 3: Response Quality
# ==============================================================================

def evaluate_response_quality(scenario: dict, transcript: str, use_llm: bool = False) -> dict:
    """
    Score: Does the agent's spoken response match the expected intent?
    Expected intent is taken from the last 'ai' turn in the scenario dialogue.
    """
    expected_intent = ""
    dialogue = scenario.get("dialogue", [])
    if dialogue:
        expected_intent = dialogue[-1].get("ai", "")

    if not use_llm:
        # Without LLM, we cannot evaluate semantics — skip gracefully
        return {"score": None, "explanation": "LLM judge disabled; response accuracy skipped."}

    score, explanation = llm_judge_response(expected_intent, transcript)
    return {"score": score, "explanation": explanation}


# ==============================================================================
# Metric 4: Latency & Interruption
# ==============================================================================

def evaluate_latency(result_data: Optional[dict]) -> dict:
    """
    Compute per-sample latency from the result JSON.

    Fields used:
      user_speech_end_rel      : float (seconds, relative to audio start)
      audio_agent_speech_start : float (seconds, relative to audio start)

    latency = audio_agent_speech_start - user_speech_end_rel
    < 0  => model started speaking BEFORE user finished => interruption
    """
    if result_data is None:
        return {"available": False}

    user_end   = result_data.get("user_speech_end_rel")
    agent_start = result_data.get("audio_agent_speech_start")

    if user_end is None or agent_start is None:
        return {"available": False, "reason": "Missing timing fields"}

    latency = round(agent_start - user_end, 3)
    is_interruption = latency < 0

    return {
        "available":             True,
        "user_speech_end_rel":   user_end,
        "audio_agent_speech_start": agent_start,
        "agent_response_latency_s": latency,
        "is_interruption":       is_interruption,
    }


# ==============================================================================
# Per-Scenario Evaluation
# ==============================================================================

def evaluate_scenario(
    scenario: dict,
    actual_calls: List[dict],
    transcript: str = "",
    result_data: Optional[dict] = None,
    use_llm: bool = False,
) -> dict:
    """Run all four metrics on a single scenario."""
    expected_calls = scenario["expected_tool_calls"]

    # Turn-taking success: did the agent produce any speech output?
    asr_chunks = []
    if result_data:
        asr_chunks = result_data.get("asr_chunks", [])
    turn_take_success = bool(transcript.strip()) or bool(asr_chunks)

    tool_sel  = evaluate_tool_selection(expected_calls, actual_calls)
    arg_acc   = evaluate_argument_accuracy(expected_calls, actual_calls, use_llm=use_llm)
    resp_qual = evaluate_response_quality(scenario, transcript, use_llm=use_llm)
    lat       = evaluate_latency(result_data)

    return {
        "scenario_id":          scenario["id"],
        "domain":               scenario["domain"],
        "difficulty":           scenario["difficulty"],
        "title":                scenario["title"],
        "turn_take_success":    turn_take_success,
        "metrics": {
            "tool_selection_acc": tool_sel,
            "argument_acc":       arg_acc,
            "response_qual":      resp_qual,
        },
        "latency":              lat,
        "expected_calls_count": len(expected_calls),
        "actual_calls_count":   len(actual_calls),
    }
