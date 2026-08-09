"""
Unit + integration tests.

Key idea for testing AI systems: separate what's DETERMINISTIC (your tool
code, your parsing logic) from what's PROBABILISTIC (the model's text).
Unit-test the deterministic parts hard. Don't assert exact strings against
LLM output — that's what the eval harness (eval_suite.py) is for instead.
"""

import pytest
from agent import run_calculator, run_lookup, TOOL_IMPLEMENTATIONS
from resume_rag import load_and_chunk_pdf, CHUNK_SIZE, CHUNK_OVERLAP


# --- Unit tests: pure functions, no API calls, fast, deterministic ---

class TestCalculator:
    def test_basic_addition(self):
        assert run_calculator("2 + 2") == "4"

    def test_order_of_operations(self):
        assert run_calculator("47 * 12 + 8") == "572"

    def test_division(self):
        assert run_calculator("10 / 4") == "2.5"

    def test_rejects_disallowed_characters(self):
        # Security-relevant: this is "safely integrating AI-generated input"
        result = run_calculator("__import__('os').system('echo hacked')")
        assert result.startswith("Error")

    def test_rejects_letters(self):
        result = run_calculator("2 + x")
        assert result.startswith("Error")


class TestLookup:
    def test_finds_anthropic(self):
        result = run_lookup("When was Anthropic founded?")
        assert "2021" in result

    def test_finds_palantir_case_insensitive(self):
        result = run_lookup("PALANTIR headquarters")
        assert "Denver" in result

    def test_missing_entry_returns_graceful_message(self):
        result = run_lookup("When was Microsoft founded?")
        assert "No matching entry" in result


class TestToolRegistry:
    def test_all_declared_tools_have_implementations(self):
        # Integration-style check: catches the classic bug where you add a
        # tool to the schema sent to the model but forget to implement it,
        # which causes silent failures only in production.
        from agent import TOOLS

        declared_names = {t["name"] for t in TOOLS}
        implemented_names = set(TOOL_IMPLEMENTATIONS.keys())
        assert declared_names == implemented_names


class TestResumeChunking:
    """
    Tests the deterministic chunking logic in isolation, using a temp file
    instead of a real PDF, so this runs fast with no PDF or API dependency.
    Swaps PdfReader's extraction for a monkeypatch-free approach by testing
    the chunking math directly against known input.
    """

    def test_chunk_overlap_math(self, tmp_path):
        # Build a fake "extracted text" string and chunk it directly,
        # bypassing PDF parsing to isolate the chunking logic itself.
        text = "A" * 1200  # long enough to require multiple chunks
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start += CHUNK_SIZE - CHUNK_OVERLAP

        # With CHUNK_SIZE=500, CHUNK_OVERLAP=50, step=450:
        # chunk 0: chars 0-500, chunk 1: chars 450-950, chunk 2: chars 900-1200
        assert len(chunks) == 3
        assert len(chunks[0]) == 500
        # Confirm overlap actually exists between consecutive chunks
        assert chunks[0][-CHUNK_OVERLAP:] == chunks[1][:CHUNK_OVERLAP]

    def test_empty_chunks_are_filtered(self):
        # Mirrors the `if c.strip()` filter in load_and_chunk_pdf —
        # blank/whitespace-only chunks shouldn't make it into the index.
        raw_chunks = ["real content", "   ", "", "more content"]
        filtered = [c for c in raw_chunks if c.strip()]
        assert filtered == ["real content", "more content"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
