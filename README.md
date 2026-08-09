# Agentic Assistant Exercise

A minimal project mapping directly to the "Applied AI Engineer" job requirements.

## Files

| File | What it teaches | Job requirement it hits |
|---|---|---|
| `agent.py` | Anthropic API tool use + the agent loop | OpenAI/Anthropic API, AI agents, prompt design, context management, multi-step workflows |
| `test_agent.py` | Unit tests on deterministic logic | Unit testing, safely validating AI-adjacent code |
| `eval_suite.py` | Scoring non-deterministic LLM output | Evaluation frameworks for AI outputs, AI-driven testing/reliability |

## How to run it

```bash
# 1. Unit tests — fast, free, no API key needed
pip install anthropic pytest
python3 -m pytest test_agent.py -v

# 2. The agent itself — needs an API key, costs a few cents
export ANTHROPIC_API_KEY="sk-ant-..."
python3 agent.py

# 3. The eval suite — runs 4 labeled cases against the live agent
python3 eval_suite.py
```

## Why this structure, not just a chatbot script

The job wants someone who treats AI output as **a system to be engineered**,
not a black box you hope works. Concretely that means:

1. **Separate deterministic code from probabilistic code.** Your tool
   functions (`run_calculator`, `run_lookup`) are pure functions — test them
   like any other code, with exact assertions, no API calls, in milliseconds.
2. **Never assert exact strings on LLM output.** The eval suite checks
   *properties* ("does the answer contain the right fact / call the right
   tool / admit uncertainty") instead of exact wording, because wording
   varies between runs even at temperature 0 in practice.
3. **Name failure modes, not just pass/fail.** Each eval case states what a
   failure *means* (hallucination vs. wrong tool vs. arithmetic error) so a
   failing run tells you what to fix, not just that something broke.
4. **The agent loop is the core mechanic.** Strip away LangChain and this is
   maybe 40 lines: call model → if it wants a tool, run it and feed the
   result back → repeat → return final text. Once you've built this raw
   once, every agent framework (LangGraph, CrewAI, etc.) is just sugar on
   top of it — which is exactly the kind of "I understand what's underneath
   the abstraction" signal this job is testing for.

## Where to take it next (closes remaining gaps fast)

- **Swap the calculator/lookup for something real** — wire `lookup` to your
  existing ChromaDB resume index instead of the fake dict. Now it's RAG +
  agent in one project, and directly reuses your existing portfolio piece.
- **Add a `web_search` tool** using the `requests` library against a real
  API. Now you have 3 tools and the model has to choose between them —
  this is what "navigate ambiguity" in the posting is actually testing.
- **Add latency/cost logging** around each `client.messages.create` call
  (track tokens used, wall-clock time per step). That's the "optimizing
  performance (latency, cost)" bullet, made concrete.
- **Push to GitHub as its own repo** with this README. It's small,
  finishable in one sitting, and demonstrates 6 of the ~10 hard
  requirements on that posting — strong portfolio addition next to the
  RAG chatbot.
