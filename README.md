# agent-eval-harness

# 

# A tool-calling agent built without a framework, paired with an eval suite that scores non-deterministic output on properties rather than exact strings.

# 

# The agent loop is roughly forty lines: call the model, run any tool it requests, feed the result back, repeat until it returns text. No LangChain, no LangGraph. The point is to show what those frameworks are wrapping.

# 

# Why it's built this way

# 

# Most agent projects are a chatbot script with tools bolted on, and they have no way to tell you whether a change made things better or worse. Three decisions address that:

# 

# Deterministic code is separated from probabilistic code. The tool functions (run\_calculator, run\_lookup) are pure functions with no API calls. They're unit-tested with exact assertions and run in milliseconds. Only the model's behaviour goes through the eval suite.

# 

# Evals assert properties, not strings. Checking that output matches expected text fails constantly — wording varies between runs even at temperature 0. The suite checks whether the answer contains the right fact, whether the correct tool was selected, and whether the model admits uncertainty instead of inventing an answer.

# 

# Each eval case names its failure mode. A case doesn't just pass or fail; it distinguishes hallucination from wrong-tool-selection from arithmetic error. A red run tells you what to fix, not just that something broke.

# 

# Structure

# File	Purpose

# agent.py	Tool definitions and the agent loop

# test\_agent.py	Unit tests on tool functions — deterministic, no API key required

# eval\_suite.py	Four labeled cases scored against the live agent

# Running it

# bash

# pip install anthropic pytest

# 

# \# Unit tests — fast, free, no API key

# python -m pytest test\_agent.py -v

# 

# \# The agent — requires an API key, costs a few cents per run

# cp .env.example .env    # add your key

# python agent.py

# 

# \# Eval suite — four cases against the live agent

# python eval\_suite.py

# Current limitations

# Two tools is a low bar for routing. With only a calculator and a lookup, tool selection is close to trivial. A third tool with overlapping scope would make the routing decision meaningful and the evals more informative.

# run\_lookup reads from a static dictionary, not a real index. Pointing it at a vector store would make retrieval quality something the evals could actually measure.

# No latency or cost instrumentation. Token counts and wall-clock time per step aren't tracked, so there's no way to catch a change that improves accuracy while tripling cost.

# Stack

# 

# Python · Anthropic API · pytest

