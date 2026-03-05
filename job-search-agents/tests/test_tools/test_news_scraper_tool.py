"""
Tests for NewsScraperTool — RSS-based funding news fetcher.
Verifies fetch error handling, RSS parsing, and funding pattern filtering.
"""

import pytest
import httpx
import respx
from tenacity import RetryError

from tools.news_scraper_tool import NewsScraperTool, RSS_SOURCES


_SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>TechCrunch VC</title>
    <item>
      <title>Acme Corp raises $10M Series A to expand AI platform</title>
      <link>https://techcrunch.com/acme-series-a</link>
      <description>Acme Corp has raised $10M in a Series A funding round led by Sequoia Capital.</description>
      <pubDate>Tue, 03 Mar 2026 10:00:00 +0000</pubDate>
    </item>
    <item>
      <title>General blog post about cloud computing</title>
      <link>https://techcrunch.com/cloud-article</link>
      <description>Cloud computing is changing the way companies build software.</description>
      <pubDate>Tue, 03 Mar 2026 09:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>"""

_EMPTY_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Empty</title></channel></rss>"""


class TestNewsScraperFetch:
    """Tests for _fetch() — HTTP error classification and retry behaviour."""

    @respx.mock
    def test_403_returns_none_without_retry(self):
        """A 403 Forbidden is permanent — should return None on first attempt, no retry."""
        url = "https://techcrunch.com/category/venture/feed/"
        respx.get(url).mock(return_value=httpx.Response(403))
        tool = NewsScraperTool(timeout=5.0)
        result = tool._fetch(url)
        assert result is None

    @respx.mock
    def test_401_returns_none_without_retry(self):
        url = "https://techcrunch.com/category/venture/feed/"
        respx.get(url).mock(return_value=httpx.Response(401))
        tool = NewsScraperTool(timeout=5.0)
        result = tool._fetch(url)
        assert result is None

    @respx.mock
    def test_404_returns_none_without_retry(self):
        url = "https://venturebeat.com/category/business/feed/"
        respx.get(url).mock(return_value=httpx.Response(404))
        tool = NewsScraperTool(timeout=5.0)
        result = tool._fetch(url)
        assert result is None

    @respx.mock
    def test_200_returns_parsed_xml(self):
        url = RSS_SOURCES["techcrunch_vc"]
        respx.get(url).mock(
            return_value=httpx.Response(
                200,
                content=_SAMPLE_RSS.encode(),
                headers={"content-type": "application/rss+xml"},
            )
        )
        tool = NewsScraperTool(timeout=5.0)
        result = tool._fetch(url)
        assert result is not None

    @respx.mock
    def test_500_raises_after_retries_exhausted(self):
        """5xx errors trigger retries; after all retries, a RetryError propagates."""
        url = RSS_SOURCES["venturebeat"]
        respx.get(url).mock(return_value=httpx.Response(500))
        tool = NewsScraperTool(timeout=5.0)
        with pytest.raises(RetryError):
            tool._fetch(url)

    @respx.mock
    def test_410_returns_none_without_retry(self):
        """410 Gone is permanent — should return None immediately without retry."""
        url = "https://example.com/old-feed/"
        respx.get(url).mock(return_value=httpx.Response(410))
        tool = NewsScraperTool(timeout=5.0)
        result = tool._fetch(url)
        assert result is None


class TestNewsScraperSearchFundingArticles:
    """Tests for search_funding_articles() — RSS end-to-end with mocked HTTP."""

    @respx.mock
    def test_returns_empty_when_source_returns_403(self):
        """If the RSS feed is blocked, return empty list — not an exception."""
        rss_url = RSS_SOURCES["techcrunch_vc"]
        respx.get(rss_url).mock(return_value=httpx.Response(403))
        tool = NewsScraperTool(timeout=5.0)
        articles = tool.search_funding_articles(
            keywords=["Series A"], sources=["techcrunch_vc"]
        )
        assert articles == []

    @respx.mock
    def test_returns_funding_articles_from_rss_feed(self):
        """RSS items matching funding patterns are returned as NewsArticle objects."""
        rss_url = RSS_SOURCES["techcrunch_vc"]
        respx.get(rss_url).mock(
            return_value=httpx.Response(
                200,
                content=_SAMPLE_RSS.encode(),
                headers={"content-type": "application/rss+xml"},
            )
        )
        tool = NewsScraperTool(timeout=5.0)
        articles = tool.search_funding_articles(
            keywords=[], sources=["techcrunch_vc"]
        )
        assert len(articles) == 1
        assert "Acme" in articles[0].title
        assert articles[0].source == "techcrunch_vc"

    @respx.mock
    def test_non_funding_articles_are_filtered_out(self):
        """Articles without funding signals are excluded from results."""
        rss_url = RSS_SOURCES["techcrunch_vc"]
        respx.get(rss_url).mock(
            return_value=httpx.Response(
                200,
                content=_SAMPLE_RSS.encode(),
                headers={"content-type": "application/rss+xml"},
            )
        )
        tool = NewsScraperTool(timeout=5.0)
        articles = tool.search_funding_articles(
            keywords=[], sources=["techcrunch_vc"]
        )
        # General cloud post should not be included
        titles = [a.title for a in articles]
        assert not any("cloud computing" in t.lower() for t in titles)

    @respx.mock
    def test_keyword_filter_further_narrows_results(self):
        """When keywords are provided, only articles matching them are returned."""
        rss_url = RSS_SOURCES["techcrunch_vc"]
        respx.get(rss_url).mock(
            return_value=httpx.Response(
                200,
                content=_SAMPLE_RSS.encode(),
                headers={"content-type": "application/rss+xml"},
            )
        )
        tool = NewsScraperTool(timeout=5.0)
        # Filter to only articles mentioning "Acme"
        articles_acme = tool.search_funding_articles(
            keywords=["Acme"], sources=["techcrunch_vc"]
        )
        articles_nomatch = tool.search_funding_articles(
            keywords=["NonExistentCompanyXYZ"], sources=["techcrunch_vc"]
        )
        assert len(articles_acme) == 1
        assert len(articles_nomatch) == 0

    @respx.mock
    def test_empty_rss_feed_returns_empty_list(self):
        rss_url = RSS_SOURCES["venturebeat"]
        respx.get(rss_url).mock(
            return_value=httpx.Response(
                200,
                content=_EMPTY_RSS.encode(),
                headers={"content-type": "application/rss+xml"},
            )
        )
        tool = NewsScraperTool(timeout=5.0)
        articles = tool.search_funding_articles(keywords=[], sources=["venturebeat"])
        assert articles == []

    def test_unknown_source_returns_empty_with_warning(self):
        """Requesting a source name not in RSS_SOURCES returns empty list gracefully."""
        tool = NewsScraperTool(timeout=5.0)
        articles = tool.search_funding_articles(
            keywords=[], sources=["nonexistent_source"]
        )
        assert articles == []
