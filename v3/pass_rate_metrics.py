#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pass_rate_metrics.py

Binary pass/fail scoring and report aggregation for evaluate_pass_rate.py.

  PASS (1) = ALL expected tools called (recall = 1.0) AND no unexpected tools
             (precision = 1.0) AND ALL arguments semantically correct.
  FAIL (0) = anything less.
"""

from typing import List, Dict, Optional
from datetime import datetime

from llm_judge import llm_judge_argument, exact_match_args


# ==============================================================================
# Per-Scenario Binary Pass/Fail Evaluation
# ==============================================================================

def evaluate_scenario_pass(
    scenario: dict,
    actual_calls: List[dict],
    transcript: str = "",
    result_data: Optional[dict] = None,
    use_llm: bool = False,
) -> dict:
    """
    Evaluate whether a scenario PASSED (all conditions met) or FAILED.

    Returns a dict with:
      - passed: bool
      - checks: dict of individual check results
      - failure_reason: str (first failing check, or "")
    """
    expected_calls = scenario["expected_tool_calls"]
    checks = {}

    # ── Check 1: Tool Selection (strict: recall=1 AND precision=1) ──
    expected_names = [c["function"] for c in expected_calls]
    actual_names = [c["function"] for c in actual_calls]

    # Multiset matching
    exp_remaining = list(expected_names)
    act_remaining = list(actual_names)
    matched = 0
    for fn in list(exp_remaining):
        if fn in act_remaining:
            matched += 1
            exp_remaining.remove(fn)
            act_remaining.remove(fn)

    recall = matched == len(expected_names)  # all expected were called
    precision = len(act_remaining) == 0       # no extra unexpected calls

    checks["tool_selection"] = {
        "passed": recall and precision,
        "expected": expected_names,
        "actual": actual_names,
        "missing": exp_remaining,
        "unexpected": act_remaining,
    }
    if not (recall and precision):
        reasons = []
        if exp_remaining:
            reasons.append(f"Missing tools: {exp_remaining}")
        if act_remaining:
            reasons.append(f"Unexpected tools: {act_remaining}")
        return _result(False, checks, "; ".join(reasons))

    # ── Check 2: Argument Accuracy (ALL arguments must match) ──
    actual_by_func: Dict[str, List[dict]] = {}
    for ac in actual_calls:
        actual_by_func.setdefault(ac["function"], []).append(ac)

    all_args_correct = True
    arg_details = []
    for ec in expected_calls:
        func = ec["function"]
        expected_args = ec.get("args", {})

        if func not in actual_by_func or not actual_by_func[func]:
            all_args_correct = False
            arg_details.append({"function": func, "passed": False, "reason": "Not called"})
            continue

        actual_call = actual_by_func[func].pop(0)
        actual_args = actual_call.get("args", {})

        if use_llm:
            is_ok, explanation = llm_judge_argument(expected_args, actual_args, func)
        else:
            is_ok, explanation = exact_match_args(expected_args, actual_args)

        if not is_ok:
            all_args_correct = False
        arg_details.append({
            "function": func,
            "passed": is_ok,
            "expected_args": expected_args,
            "actual_args": actual_args,
            "explanation": explanation,
        })

    checks["argument_accuracy"] = {"passed": all_args_correct, "details": arg_details}
    if not all_args_correct:
        failed_fns = [d["function"] for d in arg_details if not d["passed"]]
        return _result(False, checks, f"Wrong arguments for: {failed_fns}")

    # ── All checks passed ──
    return _result(True, checks, "")


def _result(passed: bool, checks: dict, failure_reason: str) -> dict:
    return {
        "passed": passed,
        "checks": checks,
        "failure_reason": failure_reason,
    }


# ==============================================================================
# Aggregate Report
# ==============================================================================

def evaluate_all_pass_rate(
    benchmark_data: dict,
    evaluation_entries: list,
    use_llm: bool = False,
) -> dict:
    """Evaluate all scenarios and produce a pass-rate report."""
    results = []

    for entry in evaluation_entries:
        scenario     = entry["scenario"]
        actual_calls = entry["calls"]
        transcript   = entry["transcript"]
        result_data  = entry["result_data"]

        eval_result = evaluate_scenario_pass(
            scenario, actual_calls,
            transcript=transcript,
            result_data=result_data,
            use_llm=use_llm,
        )

        results.append({
            "scenario_id":    scenario["id"],
            "domain":         scenario["domain"],
            "difficulty":     scenario["difficulty"],
            "title":          scenario["title"],
            "num_tools":      len(scenario["expected_tool_calls"]),
            "disfluency":     scenario.get("disfluency_features", []),
            "state_rollback": scenario.get("state_rollback_test", False),
            **eval_result,
        })

    # ── Aggregate metrics ──
    total = len(results)
    passed_list = [r for r in results if r["passed"]]
    failed_list = [r for r in results if not r["passed"]]

    pass_rate = round(len(passed_list) / total, 3) if total else 0

    # ── Failure breakdown ──
    failure_categories = {
        "wrong_tools": 0,
        "wrong_arguments": 0,
    }
    for r in failed_list:
        reason = r["failure_reason"].lower()
        if "missing tools" in reason or "unexpected tools" in reason:
            failure_categories["wrong_tools"] += 1
        elif "wrong arguments" in reason:
            failure_categories["wrong_arguments"] += 1

    # ── By domain ──
    by_domain = {}
    for r in results:
        by_domain.setdefault(r["domain"], {"total": 0, "passed": 0})
        by_domain[r["domain"]]["total"] += 1
        if r["passed"]:
            by_domain[r["domain"]]["passed"] += 1
    domain_pass_rates = {
        d: round(v["passed"] / v["total"], 3) if v["total"] else 0
        for d, v in by_domain.items()
    }

    # ── By difficulty ──
    by_difficulty = {}
    for r in results:
        by_difficulty.setdefault(r["difficulty"], {"total": 0, "passed": 0})
        by_difficulty[r["difficulty"]]["total"] += 1
        if r["passed"]:
            by_difficulty[r["difficulty"]]["passed"] += 1
    difficulty_pass_rates = {
        d: round(v["passed"] / v["total"], 3) if v["total"] else 0
        for d, v in by_difficulty.items()
    }

    # ── By num_tool_calls ──
    by_num_tools = {}
    for r in results:
        k = r["num_tools"]
        by_num_tools.setdefault(k, {"total": 0, "passed": 0})
        by_num_tools[k]["total"] += 1
        if r["passed"]:
            by_num_tools[k]["passed"] += 1
    num_tools_pass_rates = {
        str(k): round(v["passed"] / v["total"], 3) if v["total"] else 0
        for k, v in sorted(by_num_tools.items())
    }

    # ── By disfluency feature ──
    by_feature = {}
    for r in results:
        for feat in r.get("disfluency", []):
            by_feature.setdefault(feat, {"total": 0, "passed": 0})
            by_feature[feat]["total"] += 1
            if r["passed"]:
                by_feature[feat]["passed"] += 1
    feature_pass_rates = {
        f: round(v["passed"] / v["total"], 3) if v["total"] else 0
        for f, v in by_feature.items()
    }

    # ── By state_rollback ──
    rollback_scenarios = [r for r in results if r.get("state_rollback")]
    no_rollback_scenarios = [r for r in results if not r.get("state_rollback")]
    rollback_pass_rate = (
        round(sum(1 for r in rollback_scenarios if r["passed"]) / len(rollback_scenarios), 3)
        if rollback_scenarios else None
    )
    no_rollback_pass_rate = (
        round(sum(1 for r in no_rollback_scenarios if r["passed"]) / len(no_rollback_scenarios), 3)
        if no_rollback_scenarios else None
    )

    report = {
        "benchmark_name":   benchmark_data.get("benchmark_name", ""),
        "evaluated_at":     datetime.now().isoformat(),
        "total_scenarios":  total,
        "overall_pass_rate": pass_rate,
        "passed":           len(passed_list),
        "failed":           len(failed_list),
        "failure_breakdown": failure_categories,
        "by_domain":        domain_pass_rates,
        "by_difficulty":    difficulty_pass_rates,
        "by_num_tools":     num_tools_pass_rates,
        "by_disfluency_feature": feature_pass_rates,
        "by_state_rollback": {
            "with_rollback": rollback_pass_rate,
            "without_rollback": no_rollback_pass_rate,
        },
        "scenario_results": results,
    }

    return report
