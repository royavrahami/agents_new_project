"""
Tests for LLMTool — mocking OpenAI to avoid real API calls.
All tests that exercise real OpenAI code paths patch settings.openai_api_key
to a non-empty value so the early-return guard does not fire.
"""

import pytest
from unittest.mock import MagicMock, patch

from tools.llm_tool import LLMTool


def _make_openai_response(content: str) -> MagicMock:
    """Build a minimal OpenAI-shaped response object."""
    return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])


class TestLLMToolComplete:
    """Tests for the complete() method."""

    def test_complete_returns_string(self):
        with patch("tools.llm_tool.settings") as mock_settings, \
             patch("tools.llm_tool.OpenAI") as mock_openai_class:
            mock_settings.openai_api_key = "fake-key"
            mock_settings.openai_model = "gpt-4o"
            mock_settings.openai_temperature = 0.3
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_openai_response("Test response")

            tool = LLMTool(model="gpt-4o")
            result = tool.complete("Hello, world!")
        assert result == "Test response"

    def test_complete_without_api_key_returns_mock(self):
        """When no API key is set, the tool returns a safe mock response."""
        tool = LLMTool()
        result = tool.complete("Test prompt")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_complete_uses_system_override(self):
        with patch("tools.llm_tool.settings") as mock_settings, \
             patch("tools.llm_tool.OpenAI") as mock_openai_class:
            mock_settings.openai_api_key = "fake-key"
            mock_settings.openai_model = "gpt-4o"
            mock_settings.openai_temperature = 0.3
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_openai_response("Response")

            tool = LLMTool()
            tool.complete("Message", system_override="Custom system prompt")

            call_kwargs = mock_client.chat.completions.create.call_args
            messages = call_kwargs.kwargs.get("messages", [])
            system_contents = [m["content"] for m in messages if m.get("role") == "system"]
            assert "Custom system prompt" in system_contents


class TestLLMToolScoreRelevance:
    """Tests for the score_relevance() method."""

    def test_score_relevance_returns_float(self):
        with patch("tools.llm_tool.settings") as mock_settings, \
             patch("tools.llm_tool.OpenAI") as mock_openai_class:
            mock_settings.openai_api_key = "fake-key"
            mock_settings.openai_model = "gpt-4o"
            mock_settings.openai_temperature = 0.3
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_openai_response("0.85")

            tool = LLMTool()
            score = tool.score_relevance("Job description text", "Candidate profile text")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_relevance_clamps_to_range(self):
        with patch("tools.llm_tool.settings") as mock_settings, \
             patch("tools.llm_tool.OpenAI") as mock_openai_class:
            mock_settings.openai_api_key = "fake-key"
            mock_settings.openai_model = "gpt-4o"
            mock_settings.openai_temperature = 0.3
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_openai_response("1.5")

            tool = LLMTool()
            score = tool.score_relevance("JD", "Profile")
        assert score == 1.0  # Clamped to max

    def test_score_relevance_handles_invalid_response(self):
        with patch("tools.llm_tool.settings") as mock_settings, \
             patch("tools.llm_tool.OpenAI") as mock_openai_class:
            mock_settings.openai_api_key = "fake-key"
            mock_settings.openai_model = "gpt-4o"
            mock_settings.openai_temperature = 0.3
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_openai_response("not a number")

            tool = LLMTool()
            score = tool.score_relevance("JD", "Profile")
        assert score == 0.5  # Default fallback


class TestLLMToolExtractCompanyName:
    """Tests for the extract_company_name() method."""

    def test_extract_valid_company_name(self):
        with patch("tools.llm_tool.settings") as mock_settings, \
             patch("tools.llm_tool.OpenAI") as mock_openai_class:
            mock_settings.openai_api_key = "fake-key"
            mock_settings.openai_model = "gpt-4o"
            mock_settings.openai_temperature = 0.3
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            mock_client.chat.completions.create.return_value = _make_openai_response("Acme Corp")

            tool = LLMTool()
            result = tool.extract_company_name("Acme Corp raises $10M Series A")
        assert result == "Acme Corp"

    def test_extract_returns_none_for_too_long_response(self):
        with patch("tools.llm_tool.settings") as mock_settings, \
             patch("tools.llm_tool.OpenAI") as mock_openai_class:
            mock_settings.openai_api_key = "fake-key"
            mock_settings.openai_model = "gpt-4o"
            mock_settings.openai_temperature = 0.3
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            long_response = "A" * 200
            mock_client.chat.completions.create.return_value = _make_openai_response(long_response)

            tool = LLMTool()
            result = tool.extract_company_name("Some text")
        assert result is None
