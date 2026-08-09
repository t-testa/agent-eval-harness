"""
Evaluation harness for the AGENT'S behavior — the part you can't unit test
because the model's exact wording varies run to run.

This is what "define evaluation strategies for AI outputs (quality,
consistency, failure modes)" means in practice. The pattern:

  1. Build a small labeled test set (input -> expected property of output)
  2. Define a checker per case (NOT exact string match — check a PROPERTY:
     did it call the right tool? does the answer contain the right fact?)
  3. Run all cases, score pass/fail, report failure modes

This costs real API calls, so it's separate from the free/fast unit tests.
"""

import re
from agent import run_agent

EVAL_CASES = [
    {
        "name": "simple_arithmetic",
        "question": "What is 23 * 4?",
        "check": lambda answer: "92" in answer,
        "failure_mode": "wrong arithmetic or didn't use calculator",
    },
    {
        "name": "multi_step_reasoning",
        "question": "What year was Anthropic founded, and is that before or after OpenAI?",
        "check": lambda answer: "2021" in answer and "after" in answer.lower(),
        "failure_mode": "failed to chain two lookups + compare correctly",
    },
    {
        "name": "no_tool_needed",
        "question": "What's the capital of France?",
        "check": lambda answer: "paris" in answer.lower(),
        "failure_mode": "over-triggered a tool call for something it should just know",
    },
    {
        "name": "graceful_unknown",
        "question": "What year was SpaceX founded according to the knowledge base?",
        # We expect it to admit it doesn't have this in the KB, not hallucinate a year
        "check": lambda answer: bool(
            re.search(
                r"(don't have|does not contain|doesn't contain|no (matching )?entry|"
                r"not (in|available)|unable|cannot confirm|can't confirm)",
                answer.lower(),
            )
        ),
        "failure_mode": "hallucinated a fact instead of admitting the KB lookup failed",
    },
]


def run_eval_suite(verbose_agent: bool = False) -> dict:
    results = []
    for case in EVAL_CASES:
        answer = run_agent(case["question"], verbose=verbose_agent)
        passed = case["check"](answer)
        results.append(
            {
                "name": case["name"],
                "question": case["question"],
                "answer": answer,
                "passed": passed,
                "failure_mode": None if passed else case["failure_mode"],
            }
        )

    total = len(results)
    passed_count = sum(r["passed"] for r in results)

    print("\n" + "=" * 60)
    print(f"EVAL RESULTS: {passed_count}/{total} passed")
    print("=" * 60)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"\n[{status}] {r['name']}")
        print(f"  Q: {r['question']}")
        print(f"  A: {r['answer']}")
        if not r["passed"]:
            print(f"  Likely failure mode: {r['failure_mode']}")

    return {"total": total, "passed": passed_count, "results": results}


if __name__ == "__main__":
    run_eval_suite(verbose_agent=False)
