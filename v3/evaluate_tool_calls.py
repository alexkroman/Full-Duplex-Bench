#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_tool_calls.py

Multi-metric evaluation engine for multi-step tool calling.
Scores agents on four clearly separated metrics (see tool_call_metrics.py):

  1. tool_selection_acc  — F1 of recall & precision over expected vs actual tool calls
  2. argument_acc        — semantic correctness of arguments (LLM judge, gpt-4o)
  3. response_qual       — spoken response quality (LLM judge, gpt-4o)
  4. latency_metrics     — agent response latency + interruption rate

Latency is derived from time-aligned fields in the result JSON:
  user_speech_end_rel    : when user finished speaking (relative seconds)
  audio_agent_speech_start : when agent first spoke (relative seconds)
  latency = audio_agent_speech_start - user_speech_end_rel
  < 0  => interruption (model spoke before user finished)

Usage:
    python evaluate_tool_calls.py --benchmark benchmark_data.json \
        --results-dir fdb_v3_data_released --provider gemini2_5 \
        --output gemini2_5_evaluation_report.json --use-llm
    python evaluate_tool_calls.py --dry-run
"""

import json
import argparse
import sys
import math
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv(".env.local")
except ImportError:
    pass

from tool_call_metrics import evaluate_scenario


# ==============================================================================
# Aggregate Report
# ==============================================================================

def evaluate_all_v2(benchmark_data: dict, evaluation_entries: list, use_llm: bool = False) -> dict:
    """Evaluate all scenarios and produce an aggregate report."""
    results   = []

    for entry in evaluation_entries:
        scenario     = entry["scenario"]
        actual_calls = entry["calls"]
        transcript   = entry["transcript"]
        result_data  = entry["result_data"]

        result = evaluate_scenario(
            scenario, actual_calls, transcript=transcript,
            result_data=result_data, use_llm=use_llm,
        )
        results.append(result)

    # ---- Turn-taking success ----
    turn_taken = [r for r in results if r["turn_take_success"]]
    turn_take_count = len(turn_taken)
    turn_take_rate = round(turn_take_count / len(results), 3) if results else None

    # ---- Aggregate per-metric (on turn-taken samples only) ----
    sel_scores  = [r["metrics"]["tool_selection_acc"]["score"] for r in turn_taken]
    arg_scores  = [r["metrics"]["argument_acc"]["score"] for r in turn_taken]
    resp_scores = [
        r["metrics"]["response_qual"]["score"]
        for r in turn_taken
        if r["metrics"]["response_qual"].get("score") is not None
    ]

    # All-samples scores (for reference)
    sel_scores_all  = [r["metrics"]["tool_selection_acc"]["score"] for r in results]
    arg_scores_all  = [r["metrics"]["argument_acc"]["score"] for r in results]

    def _avg(lst):
        return round(sum(lst) / len(lst), 3) if lst else None

    # ---- Latency aggregation (turn-taken only) ----
    latency_samples    = [r["latency"] for r in turn_taken if r["latency"].get("available")]
    interruptions      = [s for s in latency_samples if s["is_interruption"]]
    non_interruptions  = [s for s in latency_samples if not s["is_interruption"]]
    latency_values     = [s["agent_response_latency_s"] for s in non_interruptions]

    def _std(lst):
        if len(lst) < 2:
            return None
        avg = sum(lst) / len(lst)
        variance = sum((x - avg) ** 2 for x in lst) / (len(lst) - 1)
        return round(math.sqrt(variance), 3)

    latency_report = {
        "total_samples":       len(latency_samples),
        "interruption_count":  len(interruptions),
        "interruption_rate":   round(len(interruptions) / len(latency_samples), 3) if latency_samples else None,
        "avg_response_latency_s": _avg(latency_values),
        "std_response_latency_s": _std(latency_values),
        "min_latency_s":       round(min(latency_values), 3) if latency_values else None,
        "max_latency_s":       round(max(latency_values), 3) if latency_values else None,
        "note":                "avg/std/min/max exclude interruption samples; computed on turn-taken samples only",
    }

    # ---- By domain / difficulty (per-metric breakdown) ----
    domain_metrics_all = {}
    difficulty_metrics_all = {}
    for r in results:
        d = r["domain"]
        domain_metrics_all.setdefault(d, {"tool_selection_acc": [], "argument_acc": [], "response_qual": []})
        domain_metrics_all[d]["tool_selection_acc"].append(r["metrics"]["tool_selection_acc"]["score"])
        domain_metrics_all[d]["argument_acc"].append(r["metrics"]["argument_acc"]["score"])
        rq = r["metrics"]["response_qual"].get("score")
        if rq is not None:
            domain_metrics_all[d]["response_qual"].append(rq)

        diff = r["difficulty"]
        difficulty_metrics_all.setdefault(diff, {"tool_selection_acc": [], "argument_acc": [], "response_qual": []})
        difficulty_metrics_all[diff]["tool_selection_acc"].append(r["metrics"]["tool_selection_acc"]["score"])
        difficulty_metrics_all[diff]["argument_acc"].append(r["metrics"]["argument_acc"]["score"])
        if rq is not None:
            difficulty_metrics_all[diff]["response_qual"].append(rq)

    domain_metrics_tt = {}
    difficulty_metrics_tt = {}
    for r in turn_taken:
        d = r["domain"]
        domain_metrics_tt.setdefault(d, {"tool_selection_acc": [], "argument_acc": [], "response_qual": []})
        domain_metrics_tt[d]["tool_selection_acc"].append(r["metrics"]["tool_selection_acc"]["score"])
        domain_metrics_tt[d]["argument_acc"].append(r["metrics"]["argument_acc"]["score"])
        rq = r["metrics"]["response_qual"].get("score")
        if rq is not None:
            domain_metrics_tt[d]["response_qual"].append(rq)

        diff = r["difficulty"]
        difficulty_metrics_tt.setdefault(diff, {"tool_selection_acc": [], "argument_acc": [], "response_qual": []})
        difficulty_metrics_tt[diff]["tool_selection_acc"].append(r["metrics"]["tool_selection_acc"]["score"])
        difficulty_metrics_tt[diff]["argument_acc"].append(r["metrics"]["argument_acc"]["score"])
        if rq is not None:
            difficulty_metrics_tt[diff]["response_qual"].append(rq)

    def _metric_summary(metrics_dict):
        return {
            k: {m: _avg(scores) for m, scores in v.items() if scores}
            for k, v in metrics_dict.items()
        }

    report = {
        "benchmark_name":  benchmark_data["benchmark_name"],
        "evaluated_at":    datetime.now().isoformat(),
        "total_scenarios": len(results),
        "turn_taking": {
            "total":           len(results),
            "turn_taken":      turn_take_count,
            "no_response":     len(results) - turn_take_count,
            "turn_take_rate":  turn_take_rate,
        },
        "by_metric": {
            "tool_selection_acc":     _avg(sel_scores),
            "argument_acc":           _avg(arg_scores),
            "response_qual":          _avg(resp_scores) if use_llm else None,
            "tool_selection_acc_all": _avg(sel_scores_all),
            "argument_acc_all":       _avg(arg_scores_all),
            "note":                   "*_all includes no-response samples (scored 0); default metrics are turn-taken only",
        },
        "latency": latency_report,
        "by_domain": _metric_summary(domain_metrics_all),
        "by_difficulty": _metric_summary(difficulty_metrics_all),
        "by_domain_turn_taken": _metric_summary(domain_metrics_tt),
        "by_difficulty_turn_taken": _metric_summary(difficulty_metrics_tt),
        "scenario_results": results,
    }

    return report


# ==============================================================================
# Dry Run
# ==============================================================================

def run_dry_run():
    """Verify evaluation logic with hardcoded sample data."""
    print("🧪 DRY RUN — Testing evaluation logic\n")

    sample_scenario = {
        "id": "test_01",
        "domain": "travel",
        "title": "Search + Book",
        "difficulty": "easy",
        "dialogue": [
            {"user": "Hi", "ai": "Hello!"},
            {"user": "Book a flight to London on Aug 20.", "ai": "I'll search for flights to London on August 20th and book one for you."},
        ],
        "expected_tool_calls": [
            {"function": "search_flights",  "args": {"destination": "London", "date": "August 20"}},
            {"function": "book_flight",     "args": {"passenger_name": "Alice"}},
        ],
    }

    # Test 1: Perfect
    print("— Test 1: Perfect calls")
    result = evaluate_scenario(
        sample_scenario,
        [
            {"function": "search_flights", "args": {"destination": "London",  "date": "2026-08-20"}},
            {"function": "book_flight",    "args": {"passenger_name": "Alice"}},
        ],
        transcript="I'll search for flights to London on August 20th and book one for you.",
        result_data={"user_speech_end_rel": 10.0, "audio_agent_speech_start": 12.5},
    )
    _print_result(result)

    # Test 2: Missing one call
    print("\n— Test 2: Missing book_flight")
    result = evaluate_scenario(
        sample_scenario,
        [{"function": "search_flights", "args": {"destination": "London", "date": "2026-08-20"}}],
        transcript="I found some flights to London.",
        result_data={"user_speech_end_rel": 10.0, "audio_agent_speech_start": 11.0},
    )
    _print_result(result)

    # Test 3: Interruption (agent started before user finished)
    print("\n— Test 3: Interruption (agent_start < user_end)")
    result = evaluate_scenario(
        sample_scenario,
        [{"function": "search_flights", "args": {"destination": "London", "date": "2026-08-20"}}],
        transcript="Looking up flights now.",
        result_data={"user_speech_end_rel": 10.0, "audio_agent_speech_start": 8.0},
    )
    _print_result(result)

    # Test 4: Extra unexpected call
    print("\n— Test 4: Extra unexpected call")
    result = evaluate_scenario(
        sample_scenario,
        [
            {"function": "search_flights",    "args": {"destination": "London", "date": "2026-08-20"}},
            {"function": "book_flight",       "args": {"passenger_name": "Alice"}},
            {"function": "update_identity_doc", "args": {"doc_type": "passport"}},
        ],
        transcript="Done!",
        result_data=None,
    )
    _print_result(result)

    print("\n✅ Dry run complete!")


def _print_result(r: dict):
    m = r["metrics"]
    lat = r["latency"]
    print(f"    tool_selection_acc: {m['tool_selection_acc']['score']}  (recall={m['tool_selection_acc']['recall']}, precision={m['tool_selection_acc']['precision']})")
    print(f"    argument_acc:       {m['argument_acc']['score']}")
    if m['response_qual'].get('score') is not None:
        print(f"    response_qual:      {m['response_qual']['score']}")
    else:
        print(f"    response_qual:      (LLM disabled)")
    if lat.get("available"):
        tag = " ⚡ INTERRUPTION" if lat["is_interruption"] else ""
        print(f"    latency:            {lat['agent_response_latency_s']}s{tag}")
    else:
        print(f"    latency:            N/A")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Step Tool Call Evaluator")
    parser.add_argument("--benchmark",    type=str, default="benchmark_data_v2.json")
    parser.add_argument("--results-dir",  type=str, default="fdb_v3_data_released",
                        help="Root directory containing result_<provider>.json files")
    parser.add_argument("--output",       type=str, default="evaluation_report.json")
    parser.add_argument("--provider",     type=str, default="gpt_realtime")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--use-llm",      action="store_true",
                        help="Use gpt-4o as LLM judge for argument and response accuracy")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
        return

    print(f"📖 Loading benchmark data from {args.benchmark}...")
    with open(args.benchmark, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    # Store dynamic lookup for scenarios
    scenario_map = {s["id"]: s for s in benchmark.get("scenarios", [])}

    print(f"📁 Scanning {args.results_dir} for provider '{args.provider}'...")
    import pathlib
    res_path = pathlib.Path(args.results_dir)

    all_scenarios_to_evaluate = []

    if res_path.exists():
        for res_file in res_path.rglob(f"result_{args.provider}.json"):
            with open(res_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            eid = data.get("example_id")
            if not eid or eid not in scenario_map:
                continue

            all_scenarios_to_evaluate.append({
                "scenario":    scenario_map[eid],
                "calls":       data.get("actual_tool_calls", []),
                "transcript":  data.get("transcript", ""),
                "result_data": data,
            })
    else:
        print(f"❌ Results directory not found: {args.results_dir}")
        sys.exit(1)

    print(f"🔍 Found {len(all_scenarios_to_evaluate)} valid result files matching benchmark scenarios.")

    # Modified evaluate_all to take the list directly
    report = evaluate_all_v2(benchmark, all_scenarios_to_evaluate, use_llm=args.use_llm)


    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ---- Summary ----
    print(f"\n📊 EVALUATION REPORT")
    print(f"{'=' * 56}")
    tt = report["turn_taking"]
    print(f"  Scenarios evaluated : {report['total_scenarios']}")
    print(f"  Turn-Take Success   : {tt['turn_taken']}/{tt['total']} ({tt['turn_take_rate']:.1%})")
    if tt['no_response'] > 0:
        print(f"  No Response (silent): {tt['no_response']}")
    print(f"\n  By Metric (turn-taken only, N={tt['turn_taken']}):")
    bm = report["by_metric"]
    print(f"    Tool Selection Acc : {bm['tool_selection_acc']:.1%}")
    print(f"    Argument Acc       : {bm['argument_acc']:.1%}")
    if bm['response_qual'] is not None:
        print(f"    Response Qual      : {bm['response_qual']:.1%}")
    else:
        print(f"    Response Qual      : (LLM judge disabled, use --use-llm)")
    lat = report["latency"]
    print(f"\n  Latency (N={lat['total_samples']}, turn-taken only):")
    if lat['avg_response_latency_s'] is not None:
        std_str = f" ± {lat['std_response_latency_s']:.2f}s" if lat.get('std_response_latency_s') else ""
        print(f"    Avg latency        : {lat['avg_response_latency_s']:.2f}s{std_str}  (excl. interruptions)")
        print(f"    Min / Max          : {lat['min_latency_s']:.2f}s / {lat['max_latency_s']:.2f}s")
    if lat['interruption_rate'] is not None:
        print(f"    Interruptions      : {lat['interruption_count']} / {lat['total_samples']}  ({lat['interruption_rate']:.1%})")
    print(f"\n  By Domain (all samples):")
    for d, metrics in report["by_domain"].items():
        parts = [f"{m}={v:.1%}" for m, v in metrics.items()]
        print(f"    {d}: {', '.join(parts)}")
    print(f"\n  By Difficulty (all samples):")
    for d, metrics in report["by_difficulty"].items():
        parts = [f"{m}={v:.1%}" for m, v in metrics.items()]
        print(f"    {d}: {', '.join(parts)}")
    print(f"\n  📄 Full report saved: {args.output}")


if __name__ == "__main__":
    main()
