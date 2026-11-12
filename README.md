# Agent Eval Harness

A small tool-calling agent built directly on the Anthropic SDK, with deterministic tool tests and a separate four-case live evaluation suite.

## How it works

The agent sends a question to the model, executes requested tools, returns tool results and repeats for up to five steps. Three tools are registered:

| Tool | Implementation |
|---|---|
| Calculator | Character-restricted arithmetic evaluation |
| Lookup | A small static company-facts dictionary |
| Resume lookup | PDF text chunks, OpenAI embeddings and an in-memory Chroma index |

The evaluation suite checks answer properties: arithmetic, a founding-year comparison, a known fact and acknowledgment of missing knowledge-base information. It checks final answer text, **not the actual tool-call trace**, so a pass does not prove that the correct tool was used.

## Setup

From the repository root, in a Python virtual environment:

```powershell
python -m pip install anthropic python-dotenv pypdf chromadb openai pytest
Copy-Item .env.example .env
```

Set `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` locally. For resume questions, supply your own PDF and update `PDF_PATH` in `resume_rag.py`; the configured resume file is not committed. PDF chunks are sent to OpenAI for embedding and retrieved text is passed to the agent.

## Run

```powershell
python agent.py
python eval_suite.py
```

Both commands call a live model and incur API usage. Resume retrieval additionally calls the embedding API.

## Unit tests without model calls

The current modules create API clients during import, so the test process still needs nonempty key values. For offline unit tests, temporary dummy values are sufficient:

```powershell
$env:ANTHROPIC_API_KEY = "unit-test-placeholder"
$env:OPENAI_API_KEY = "unit-test-placeholder"
python -m pytest test_agent.py -v
Remove-Item Env:ANTHROPIC_API_KEY
Remove-Item Env:OPENAI_API_KEY
```

Use a separate test terminal so these temporary values do not replace credentials in your live-run terminal. The checked-in tests exercise calculator behavior, dictionary lookup, tool registration and chunking assumptions; they do not call the model.

## Files

| File | Purpose |
|---|---|
| `agent.py` | Tool definitions and agent loop |
| `resume_rag.py` | PDF retrieval tool |
| `test_agent.py` | Deterministic tests |
| `eval_suite.py` | Four live answer-property checks |

## Limitations

This is a learning prototype, with a tiny evaluation set, static lookup data and no token/cost instrumentation. The calculator's character filter is not a resource-use sandbox. Dependencies are not pinned in a requirements file. Resume retrieval is not covered by the four live evaluation cases, and the suite's printed results are not a production release gate.
