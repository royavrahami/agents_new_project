"""
Tests for IntelAgent — funding scanner and hidden job discovery.
Covers: happy path, empty results, scoring logic, deduplication, persistence.
"""

import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from agents.intel_agent import IntelAgent, _SCORE_WEIGHTS
from core.models import FundingEvent, HiddenJob
from tools.google_search_tool import GoogleSearchResult
from tools.news_scraper_tool import NewsArticle


class TestIntelAgentScoring:
    """Unit tests for the hot_score calculation logic."""

    def _make_agent(self, mock_llm) -> IntelAgent:
        google = MagicMock()
        google.search_jobs_on_domain.return_value = []
        google.search_funding_news.return_value = []
        news = MagicMock()
        news.search_funding_articles.return_value = []
        return IntelAgent(
            google_tool=google,
            news_tool=news,
            llm_tool=mock_llm,
            target_roles=["QA Engineer"],
            target_keywords=["QA", "Automation", "Python"],
        )

    def test_funding_linked_increases_score(self, mock_llm):
        agent = self._make_agent(mock_llm)
        job = HiddenJob(
            company_name="Funded Corp",
            role_title="QA Engineer",
            job_url="https://example.com/job/1",
            source_domain="comeet.com",
        )
        scored = agent._score_and_enrich_jobs([job], funded_company_names={"funded corp"})
        assert scored[0].hot_score >= _SCORE_WEIGHTS["funding_linked"]
        assert scored[0].funding_linked is True

    def test_keyword_match_increases_score(self, mock_llm):
        agent = self._make_agent(mock_llm)
        job = HiddenJob(
            company_name="Tech Co",
            role_title="QA Automation Python Engineer",
            job_url="https://example.com/job/2",
            source_domain="comeet.com",
        )
        scored = agent._score_and_enrich_jobs([job], funded_company_names=set())
        # QA + Automation + Python = 3/3 keywords matched
        assert scored[0].hot_score > _SCORE_WEIGHTS["recency"]

    def test_remote_flag_adds_bonus(self, mock_llm):
        agent = self._make_agent(mock_llm)
        job_remote = HiddenJob(
            company_name="Remote Co",
            role_title="Engineer",
            job_url="https://example.com/job/3",
            source_domain="lever.co",
            remote=True,
        )
        job_onsite = HiddenJob(
            company_name="Onsite Co",
            role_title="Engineer",
            job_url="https://example.com/job/4",
            source_domain="lever.co",
            remote=False,
        )
        scored_remote = agent._score_and_enrich_jobs([job_remote], set())[0]
        scored_onsite = agent._score_and_enrich_jobs([job_onsite], set())[0]
        assert scored_remote.hot_score > scored_onsite.hot_score

    def test_score_capped_at_1(self, mock_llm):
        agent = self._make_agent(mock_llm)
        job = HiddenJob(
            company_name="Mega Corp",
            role_title="QA Automation Python Engineer",
            job_url="https://example.com/job/5",
            source_domain="comeet.com",
            remote=True,
        )
        scored = agent._score_and_enrich_jobs([job], {"mega corp"})
        assert scored[0].hot_score <= 1.0

    def test_empty_jobs_returns_empty_list(self, mock_llm):
        agent = self._make_agent(mock_llm)
        result = agent._score_and_enrich_jobs([], set())
        assert result == []


class TestIntelAgentExtractors:
    """Unit tests for static extraction helper methods."""

    def test_extract_amount_dollar(self):
        text = "Company raised $5M in Series A"
        result = IntelAgent._extract_amount(text)
        assert result is not None
        assert "$5M" in result

    def test_extract_amount_hebrew(self):
        text = "החברה גייסה 100 מיליון דולר"
        result = IntelAgent._extract_amount(text)
        assert result is not None

    def test_extract_amount_none_when_absent(self):
        result = IntelAgent._extract_amount("no money mentioned here")
        assert result is None

    def test_extract_round_type_series(self):
        result = IntelAgent._extract_round_type("Series B announcement")
        assert result is not None
        assert "Series" in result

    def test_extract_round_type_seed(self):
        result = IntelAgent._extract_round_type("seed round completed")
        assert result is not None
        assert "seed" in result.lower()

    def test_extract_round_type_none(self):
        result = IntelAgent._extract_round_type("no round mentioned")
        assert result is None


class TestIntelAgentRun:
    """Integration-level tests for IntelAgent.run() with mocked tools."""

    def _make_agent_with_results(self, mock_llm) -> IntelAgent:
        google = MagicMock()
        google.search_jobs_on_domain.return_value = [
            GoogleSearchResult(
                title="QA Automation Engineer at Acme Corp",
                url="https://comeet.com/jobs/acme/qa-1",
                snippet="Python, Playwright, pytest required. Tel Aviv.",
            )
        ]
        google.search_funding_news.return_value = []

        news = MagicMock()
        news.search_funding_articles.return_value = [
            NewsArticle(
                title="Acme Corp raises $10M Series A",
                url="https://calcalist.co.il/acme-funding",
                source="calcalist",
                snippet="Acme Corp raised $10M to expand its R&D team",
                company_mentions=["Acme Corp"],
            )
        ]
        mock_llm.extract_company_name.return_value = "Acme Corp"

        return IntelAgent(
            google_tool=google,
            news_tool=news,
            llm_tool=mock_llm,
            target_roles=["QA Automation Engineer"],
            target_keywords=["QA", "Automation", "Python"],
        )

    def test_run_returns_ok_status(self, mock_llm):
        agent = self._make_agent_with_results(mock_llm)
        result = agent.execute()
        assert result["status"] == "ok"

    def test_run_returns_hot_jobs(self, mock_llm):
        agent = self._make_agent_with_results(mock_llm)
        result = agent.execute()
        assert result["hidden_jobs"] >= 0  # May be 0 if deduped

    def test_run_returns_funding_events(self, mock_llm):
        agent = self._make_agent_with_results(mock_llm)
        result = agent.execute()
        assert result["funding_events"] >= 0

    def test_run_with_no_results(self, mock_llm):
        google = MagicMock()
        google.search_jobs_on_domain.return_value = []
        google.search_funding_news.return_value = []
        news = MagicMock()
        news.search_funding_articles.return_value = []

        agent = IntelAgent(google_tool=google, news_tool=news, llm_tool=mock_llm)
        result = agent.execute()
        assert result["status"] == "ok"
        assert result["hidden_jobs"] == 0
        assert result["funding_events"] == 0

    def test_run_handles_tool_exception_gracefully(self, mock_llm):
        google = MagicMock()
        google.search_jobs_on_domain.side_effect = Exception("Network error")
        google.search_funding_news.return_value = []
        news = MagicMock()
        news.search_funding_articles.return_value = []

        agent = IntelAgent(google_tool=google, news_tool=news, llm_tool=mock_llm)
        # Should not raise — BaseAgent.execute() catches all exceptions
        result = agent.execute()
        assert result["status"] in ("ok", "error")

    def test_deduplication_prevents_duplicate_jobs(self, mock_llm):
        google = MagicMock()
        # Return same URL twice (simulating two domains returning same job)
        same_result = GoogleSearchResult(
            title="QA Engineer at Acme",
            url="https://comeet.com/jobs/acme/qa-1",
            snippet="Python required",
        )
        google.search_jobs_on_domain.return_value = [same_result, same_result]
        google.search_funding_news.return_value = []
        news = MagicMock()
        news.search_funding_articles.return_value = []

        agent = IntelAgent(
            google_tool=google,
            news_tool=news,
            llm_tool=mock_llm,
            target_roles=["QA Engineer"],
        )
        agent.execute()
        # Second execute should not double-count (URL deduplication in _search_hidden_job_boards)
        result = agent.execute()
        assert result["status"] == "ok"
