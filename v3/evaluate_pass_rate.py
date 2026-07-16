#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_pass_rate.py

Binary pass/fail evaluation for multi-step tool-calling voice agents.

Unlike evaluate_tool_calls.py which produces averaged continuous scores
(tool_selection_acc, argument_acc, response_acc), this script provides a
strict BINARY "task completion pass rate" metric focused purely on tool usage
(scoring logic lives in pass_rate_metrics.py):

  PASS (1) = ALL of the following conditions are met:
    1. Tool Selection      — ALL expected tools were called (recall = 1.0)
                             AND no unexpected tools were called (precision = 1.0)
    2. Argument Accuracy   — ALL arguments for ALL called tools are semantically correct
                             (LLM judge, same as evaluate_tool_calls.py)

  FAIL (0) = ANY of the above conditions is not met.

This is intentionally stricter than evaluate_tool_calls.py:
  - A scenario with 2/3 correct tool calls scores ~0.667 in tool_selection_acc,
    but scores FAIL (0) in pass rate because not ALL tools were called.
  - A scenario with all tools correct but one wrong argument scores partial
    in argument_acc, but FAIL (0) in pass rate.

Output: {provider}_pass_rate_report.json

Usage:
    python evaluate_pass_rate.py --benchmark benchmark_data.json \\
        --results-dir fdb_v3_data_released --provider gpt_realtime \\
        --output gpt_realtime_pass_rate_report.json --use-llm

    # Without LLM (uses exact match for arguments, skips response check):
    python evaluate_pass_rate.py --benchmark benchmark_data.json \\
        --results-dir fdb_v3_data_released --provider gpt_realtime

    # Dry run to verify logic:
    python evaluate_pass_rate.py --dry-run
"""

import json
import argparse
import sys
import pathlib

try:
    from dotenv import load_dotenv
    load_dotenv(".env.local")
except ImportError:
    pass

from pass_rate_metrics import evaluate_scenario_pass, evaluate_all_pass_rate


# ==============================================================================
# Dry Run
# ==============================================================================

def run_dry_run():
    """Verify evaluation logic with hardcoded sample data."""
    print("🧪 DRY RUN — Testing pass rate logic\n")

    sample_scenario = {
        "id": "test_01",
        "domain": "travel",
        "title": "Search + Book",
        "difficulty": "easy",
        "disfluency_features": ["FILLER"],
        "state_rollback_test": False,
        "dialogue": [
            {"user": "Hi", "ai": "Hello!"},
            {"user": "Book a flight to London on Aug 20.", "ai": "I'll search for flights to London on August 20th and book one for you."},
        ],
        "expected_tool_calls": [
            {"function": "search_flights",  "args": {"destination": "London", "date": "August 20"}},
            {"function": "book_flight",     "args": {"passenger_name": "Alice"}},
        ],
    }

    # Test 1: Perfect — should PASS
    print("— Test 1: Perfect calls (expect PASS)")
    result = evaluate_scenario_pass(
        sample_scenario,
        [
            {"function": "search_flights", "args": {"destination": "London",  "date": "2026-08-20"}},
            {"function": "book_flight",    "args": {"passenger_name": "Alice"}},
        ],
        transcript="I'll search for flights to London on August 20th and book one for you.",
        result_data={"asr_chunks": [{"text": "hello"}]},
    )
    _print_pass_result(result)

    # Test 2: Missing one call — should FAIL
    print("\n— Test 2: Missing book_flight (expect FAIL)")
    result = evaluate_scenario_pass(
        sample_scenario,
        [
            {"function": "search_flights", "args": {"destination": "London",  "date": "2026-08-20"}},
        ],
        transcript="Searching for flights.",
        result_data={"asr_chunks": [{"text": "hello"}]},
    )
    _print_pass_result(result)

    # Test 3: No tool calls — should FAIL
    print("\n— Test 3: No tool calls (expect FAIL)")
    result = evaluate_scenario_pass(
        sample_scenario,
        [],
        transcript="I'll help you with that.",
        result_data={"asr_chunks": [{"text": "hello"}]},
    )
    _print_pass_result(result)

    # Test 4: Extra unexpected call — should FAIL
    print("\n— Test 4: Extra unexpected call (expect FAIL)")
    result = evaluate_scenario_pass(
        sample_scenario,
        [
            {"function": "search_flights", "args": {"destination": "London", "date": "2026-08-20"}},
            {"function": "book_flight",    "args": {"passenger_name": "Alice"}},
            {"function": "cancel_flight",  "args": {"flight_id": "F123"}},
        ],
        transcript="Done!",
        result_data={"asr_chunks": [{"text": "done"}]},
    )
    _print_pass_result(result)

    # Test 5: Wrong argument — should FAIL
    print("\n— Test 5: Wrong argument (expect FAIL)")
    result = evaluate_scenario_pass(
        sample_scenario,
        [
            {"function": "search_flights", "args": {"destination": "Paris",  "date": "2026-08-20"}},
            {"function": "book_flight",    "args": {"passenger_name": "Alice"}},
        ],
        transcript="Searching for flights to Paris.",
        result_data={"asr_chunks": [{"text": "hello"}]},
    )
    _print_pass_result(result)

    print("\n✅ Dry run complete!")


def _print_pass_result(r: dict):
    status = "✅ PASS" if r["passed"] else "❌ FAIL"
    print(f"  Result: {status}")
    if r["failure_reason"]:
        print(f"  Reason: {r['failure_reason']}")
    for check_name, check_data in r["checks"].items():
        icon = "✓" if check_data.get("passed") else "✗"
        print(f"    {icon} {check_name}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Task Completion Pass Rate Evaluator")
    parser.add_argument("--benchmark",    type=str, default="benchmark_data_v2.json")
    parser.add_argument("--results-dir",  type=str, default="fdb_v3_data_released",
                        help="Root directory containing result_<provider>.json files")
    parser.add_argument("--output",       type=str, default=None,
                        help="Output file (default: {provider}_pass_rate_report.json)")
    parser.add_argument("--provider",     type=str, default="gpt_realtime")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--use-llm",      action="store_true",
                        help="Use gpt-4o as LLM judge for arguments and response quality")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
        return

    if args.output is None:
        args.output = f"{args.provider}_pass_rate_report.json"

    print(f"📖 Loading benchmark data from {args.benchmark}...")
    with open(args.benchmark, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    scenario_map = {s["id"]: s for s in benchmark.get("scenarios", [])}

    print(f"📁 Scanning {args.results_dir} for provider '{args.provider}'...")
    res_path = pathlib.Path(args.results_dir)

    all_entries = []
    if res_path.exists():
        for res_file in res_path.rglob(f"result_{args.provider}.json"):
            with open(res_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            eid = data.get("example_id")
            if not eid or eid not in scenario_map:
                continue
            all_entries.append({
                "scenario":    scenario_map[eid],
                "calls":       data.get("actual_tool_calls", []),
                "transcript":  data.get("transcript", ""),
                "result_data": data,
            })
    else:
        print(f"❌ Results directory not found: {args.results_dir}")
        sys.exit(1)

    print(f"🔍 Found {len(all_entries)} valid result files matching benchmark scenarios.")

    report = evaluate_all_pass_rate(benchmark, all_entries, use_llm=args.use_llm)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Summary ──
    print(f"\n📊 PASS RATE REPORT — {args.provider.upper()}")
    print(f"{'=' * 56}")
    print(f"  Total Scenarios : {report['total_scenarios']}")
    print(f"  ✅ Passed       : {report['passed']}")
    print(f"  ❌ Failed       : {report['failed']}")
    print(f"  📈 Pass Rate    : {report['overall_pass_rate']:.1%}")

    fb = report["failure_breakdown"]
    print(f"\n  Failure Breakdown:")
    print(f"    Wrong Tools     : {fb['wrong_tools']}")
    print(f"    Wrong Arguments : {fb['wrong_arguments']}")

    print(f"\n  By Domain:")
    for d, rate in report["by_domain"].items():
        print(f"    {d}: {rate:.1%}")

    print(f"\n  By Difficulty:")
    for d, rate in report["by_difficulty"].items():
        print(f"    {d}: {rate:.1%}")

    print(f"\n  By Num Tool Calls:")
    for k, rate in report["by_num_tools"].items():
        print(f"    {k} tools: {rate:.1%}")

    if report["by_disfluency_feature"]:
        print(f"\n  By Disfluency Feature:")
        for f, rate in report["by_disfluency_feature"].items():
            print(f"    {f}: {rate:.1%}")

    rb = report["by_state_rollback"]
    if rb["with_rollback"] is not None:
        print(f"\n  State Rollback:")
        print(f"    With rollback:    {rb['with_rollback']:.1%}")
        print(f"    Without rollback: {rb['without_rollback']:.1%}")

    print(f"\n  📄 Report saved: {args.output}")


if __name__ == "__main__":
    main()
