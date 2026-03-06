"""
Test Runner: Execute and validate the three test scenarios.
Runs the queries directly against the agent and captures full traces.
"""

import os
import sys
import json
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from etl.vector_store import VectorStore
from agent.graph import CyberIrelandAgent
from utils.logger import AgentTracer


# --- Test Scenarios ---
TEST_SCENARIOS = [
    {
        "id": "test_1_verification",
        "name": "The Verification Challenge",
        "query": "What is the total number of jobs reported, and where exactly is this stated?",
        "expected_keywords": ["7,351", "7351", "employment"],
        "validation": {
            "must_contain_number": True,
            "must_contain_page_ref": True,
            "must_contain_quote": True,
        },
    },
    {
        "id": "test_2_data_synthesis",
        "name": "The Data Synthesis Challenge",
        "query": "Compare the concentration of 'Pure-Play' cybersecurity firms in the South-West against the National Average.",
        "expected_keywords": ["dedicated", "pure-play", "cork", "south", "national"],
        "validation": {
            "must_contain_number": True,
            "must_contain_page_ref": True,
            "must_contain_comparison": True,
        },
    },
    {
        "id": "test_3_forecasting",
        "name": "The Forecasting Challenge",
        "query": "Based on our 2022 baseline and the stated 2030 job target, what is the required compound annual growth rate (CAGR) to hit that goal?",
        "expected_keywords": ["CAGR", "growth", "2030", "17,333", "7,351"],
        "validation": {
            "must_contain_number": True,
            "must_contain_page_ref": True,
            "must_contain_calculation": True,
        },
    },
]


def validate_result(result: dict, scenario: dict) -> dict:
    """Validate the result against expected criteria."""
    answer = result.get("answer", "")
    citations = result.get("citations", [])

    checks = {
        "has_answer": bool(answer and len(answer) > 50),
        "has_citations": bool(citations),
        "has_page_reference": bool(citations or "page" in answer.lower()),
    }

    # Check for expected keywords
    answer_lower = answer.lower()
    keyword_hits = 0
    for kw in scenario.get("expected_keywords", []):
        if kw.lower() in answer_lower:
            keyword_hits += 1
    checks["keyword_coverage"] = f"{keyword_hits}/{len(scenario.get('expected_keywords', []))}"

    # Check for numbers
    import re
    numbers_found = re.findall(r'\d[\d,\.]+', answer)
    checks["contains_numbers"] = bool(numbers_found)

    # Check if tool calls were made
    steps = result.get("steps", [])
    tool_calls = [s for s in steps if s.get("type") == "tool_call"]
    checks["num_tool_calls"] = len(tool_calls)
    checks["tools_used"] = list(set(s.get("tool", "") for s in tool_calls))

    # Overall pass/fail
    critical_checks = [
        checks["has_answer"],
        checks["has_citations"] or checks["has_page_reference"],
        checks["contains_numbers"],
    ]
    checks["passed"] = all(critical_checks)

    return checks


def run_test_scenarios():
    """Run all three test scenarios and generate traces."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    persist_dir = os.path.join(project_root, "chroma_db")
    log_dir = os.path.join(project_root, "logs")

    # Check prerequisites
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set. Add it to .env file.")
        sys.exit(1)

    # Initialize
    print("=" * 70)
    print("CYBER IRELAND 2022 REPORT - TEST SCENARIO RUNNER")
    print("=" * 70)

    print("\n[INIT] Loading vector store...")
    vector_store = VectorStore(persist_dir=persist_dir)
    count = vector_store.collection.count()
    if count == 0:
        print("ERROR: Vector store is empty! Run ETL pipeline first:")
        print("  python -m etl.run_pipeline")
        sys.exit(1)
    print(f"[INIT] Vector store has {count} chunks.")

    print("[INIT] Initializing agent...")
    tracer = AgentTracer(log_dir=log_dir)
    agent = CyberIrelandAgent(vector_store=vector_store, tracer=tracer)
    print("[INIT] Agent ready.\n")

    # Run tests
    all_results = []
    for scenario in TEST_SCENARIOS:
        print("=" * 70)
        print(f"TEST: {scenario['name']}")
        print(f"QUERY: {scenario['query']}")
        print("=" * 70)

        start = time.time()
        result = agent.query(scenario["query"])
        duration = (time.time() - start) * 1000

        # Validate
        validation = validate_result(result, scenario)

        print(f"\n{'='*70}")
        print(f"RESULT for {scenario['name']}:")
        print(f"{'='*70}")
        print(f"Duration: {duration:.0f}ms")
        print(f"Trace ID: {result['trace_id']}")
        print(f"Citations: {result['citations']}")
        print(f"Validation: {'PASSED' if validation['passed'] else 'FAILED'}")
        for k, v in validation.items():
            print(f"  {k}: {v}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\n{'='*70}\n")

        all_results.append({
            "scenario": scenario,
            "result": {
                "answer": result["answer"],
                "citations": result["citations"],
                "trace_id": result["trace_id"],
                "steps": result["steps"],
                "duration_ms": round(duration, 2),
            },
            "validation": validation,
        })

        # Delay between tests (avoid rate limiting on free tier)
        print("[WAIT] Pausing 15s between tests to respect rate limits...")
        time.sleep(15)

    # Save all results
    output_file = os.path.join(log_dir, "test_scenario_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[DONE] Results saved to {output_file}")

    # Save all traces
    tracer.save_all_traces("test_scenario_traces.json")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in all_results:
        status = "✓ PASSED" if r["validation"]["passed"] else "✗ FAILED"
        print(f"  {status} - {r['scenario']['name']}")
        print(f"    Duration: {r['result']['duration_ms']:.0f}ms | Citations: {r['result']['citations']}")
        print(f"    Tools used: {r['validation']['tools_used']}")
    print("=" * 70)


if __name__ == "__main__":
    run_test_scenarios()
