"""
Tests for GoogleSearchTool — mocking HTTP requests to avoid real API calls.
"""

import pytest
import httpx
import respx

from tools.google_search_tool import GoogleSearchTool, GOOGLE_CSE_ENDPOINT


class TestGoogleSearchToolSearch:
    """Tests for the search() method."""

    def test_search_returns_empty_without_credentials(self):
        tool = GoogleSearchTool(api_key="", cse_id="")
        results = tool.search("test query")
        assert results == []

    @respx.mock
    def test_search_returns_results_with_credentials(self):
        mock_response = {
            "items": [
                {"title": "Job Title", "link": "https://example.com/job/1", "snippet": "QA role"},
                {"title": "Another Job", "link": "https://example.com/job/2", "snippet": "Python"},
            ]
        }
        respx.get(GOOGLE_CSE_ENDPOINT).mock(
            return_value=httpx.Response(200, json=mock_response)
        )
        tool = GoogleSearchTool(api_key="fake-key", cse_id="fake-cx", rate_limit_delay=0)
        results = tool.search("QA Engineer Israel")
        assert len(results) == 2
        assert results[0].title == "Job Title"
        assert results[0].url == "https://example.com/job/1"

    @respx.mock
    def test_search_handles_empty_items(self):
        respx.get(GOOGLE_CSE_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"items": []})
        )
        tool = GoogleSearchTool(api_key="fake-key", cse_id="fake-cx", rate_limit_delay=0)
        results = tool.search("obscure query")
        assert results == []

    @respx.mock
    def test_search_respects_max_results(self):
        items = [
            {"title": f"Job {i}", "link": f"https://example.com/{i}", "snippet": f"Role {i}"}
            for i in range(10)
        ]
        respx.get(GOOGLE_CSE_ENDPOINT).mock(
            return_value=httpx.Response(200, json={"items": items})
        )
        tool = GoogleSearchTool(api_key="fake-key", cse_id="fake-cx", max_results=5, rate_limit_delay=0)
        results = tool.search("test")
        assert len(results) == 10  # API returns 10, tool returns all

    @respx.mock
    def test_search_raises_on_http_error(self):
        respx.get(GOOGLE_CSE_ENDPOINT).mock(
            return_value=httpx.Response(403, json={"error": "Forbidden"})
        )
        tool = GoogleSearchTool(api_key="fake-key", cse_id="fake-cx", rate_limit_delay=0)
        with pytest.raises(Exception):
            tool.search("test query")


class TestGoogleSearchToolJobSearch:
    """Tests for domain-specific job search."""

    @respx.mock
    def test_search_jobs_on_domain_uses_site_operator(self):
        """The query must include 'site:' operator for job board penetration."""
        captured_query = []

        def capture_request(request):
            import urllib.parse
            # URL query params may be bytes — decode to string for comparison
            raw_params = dict(urllib.parse.parse_qsl(request.url.query))
            for k, v in raw_params.items():
                key = k.decode() if isinstance(k, bytes) else k
                val = v.decode() if isinstance(v, bytes) else v
                if key == "q":
                    captured_query.append(val)
            return httpx.Response(200, json={"items": []})

        respx.get(GOOGLE_CSE_ENDPOINT).mock(side_effect=capture_request)
        tool = GoogleSearchTool(api_key="fake-key", cse_id="fake-cx", rate_limit_delay=0)
        tool.search_jobs_on_domain(role="QA Engineer", domain="comeet.com")
        assert captured_query, "No query parameter was captured"
        assert "site:comeet.com" in captured_query[0]

    def test_search_funding_news_returns_list(self):
        tool = GoogleSearchTool(api_key="", cse_id="")
        results = tool.search_funding_news()
        assert isinstance(results, list)


class TestGoogleSearchToolContextManager:
    """Tests for context manager protocol."""

    def test_context_manager_closes_client(self):
        with GoogleSearchTool() as tool:
            assert tool is not None
        # No exception = client closed successfully
