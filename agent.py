"""
Minimal agentic assistant — demonstrates:
- Anthropic API + tool use (function calling)
- Multi-step agent loop (the model decides what to call, you execute it)
- Context management (conversation history passed back in)
- Prompt design (system prompt shapes behavior)

This intentionally has NO LangChain — at the FDE/AI-agent level, interviewers
often want to see you understand the raw mechanics underneath the framework.
"""

import os
import json
from dotenv import load_dotenv
from anthropic import Anthropic
from resume_rag import RESUME_TOOL_SCHEMA, resume_lookup

load_dotenv()  # reads .env in this folder and loads it into os.environ
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a precise research assistant with access to three tools:
calculator, lookup, and resume_lookup. Use a tool whenever the question requires
it instead of guessing. If you already know the answer with certainty and no
tool is needed, answer directly. Keep final answers to one or two sentences."""

# --- Tool definitions: this is the "context management / tool use" part ---
TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression. Use for any math.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression, e.g. '23 * 4 + 1'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "lookup",
        "description": "Look up a fact from a small local knowledge base "
        "(company founding years, headquarters). Use for factual questions "
        "covered by the knowledge base instead of guessing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look up"}
            },
            "required": ["query"],
        },
    },
    RESUME_TOOL_SCHEMA,
]

# Fake "knowledge base" so this exercise runs with zero external dependencies.
# In a real job you'd swap this for a vector DB / RAG lookup or a real API call.
KNOWLEDGE_BASE = {
    "anthropic": {"founded": 2021, "hq": "San Francisco"},
    "openai": {"founded": 2015, "hq": "San Francisco"},
    "palantir": {"founded": 2003, "hq": "Denver"},
}


def run_calculator(expression: str) -> str:
    """Real tool execution. Deliberately restricted eval for safety."""
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "Error: expression contains disallowed characters."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def run_lookup(query: str) -> str:
    q = query.lower()
    for name, facts in KNOWLEDGE_BASE.items():
        if name in q:
            return json.dumps(facts)
    return "No matching entry in knowledge base."


TOOL_IMPLEMENTATIONS = {
    "calculator": lambda inp: run_calculator(inp["expression"]),
    "lookup": lambda inp: run_lookup(inp["query"]),
    "resume_lookup": lambda inp: resume_lookup(inp["query"]),
}


def run_agent(user_question: str, max_steps: int = 5, verbose: bool = True) -> str:
    """
    The agent loop: call the model -> if it wants a tool, run the tool and
    feed the result back -> repeat until the model gives a final text answer
    or we hit max_steps. This loop IS the "agentic workflow" the job wants.
    """
    messages = [{"role": "user", "content": user_question}]

    for step in range(max_steps):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if verbose:
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    print(f"  [step {step}] model says: {block.text.strip()}")
                elif block.type == "tool_use":
                    print(f"  [step {step}] model calls: {block.name}({block.input})")

        if response.stop_reason != "tool_use":
            # Model is done — extract final text
            final_text = "".join(
                b.text for b in response.content if b.type == "text"
            )
            return final_text.strip()

        # Model wants to use one or more tools — execute and feed results back
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                impl = TOOL_IMPLEMENTATIONS.get(block.name)
                result = impl(block.input) if impl else "Error: unknown tool"
                if verbose:
                    print(f"  [step {step}] tool result: {result}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )
        messages.append({"role": "user", "content": tool_results})

    return "Error: max steps reached without a final answer."


if __name__ == "__main__":
    questions = [
        "What is 47 * 12 + 8?",
        "What year was Anthropic founded, and is that before or after OpenAI?",
        "What's the capital of France?",  # no tool needed — tests judgment
        "What companies has Tony worked for, according to his resume?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        answer = run_agent(q)
        print(f"A: {answer}")
