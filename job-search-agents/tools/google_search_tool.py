"""
Google Custom Search Tool.
Wraps the Google Custom Search JSON API for structured web queries.
Falls back to a simple requests-based scrape when no API key is configured (dev mode).

Docs: https://developers.google.com/custom-search/v1/overview
"""

from __future__ import annotations

import time
from typing import List, Optional
from urllib.parse import quote_plus

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from core.logger import logger
from core.models import HiddenJob


GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"


class GoogleSearchResult:
    """Raw result from a single Google search hit."""

    def __init__(self, title: str, url: str, snippet: str):
        self.title = title
        self.url = url
        self.snippet = snippet

    def __repr__(self) -> str:
        return f"<GoogleSearchResult title={self.title!r} url={self.url!r}>"


class GoogleSearchTool:
    """
    Thin wrapper around Google Custom Search API.

    Args:
        api_key: Google API key (defaults to settings.google_api_key)
        cse_id:  Custom Search Engine ID (defaults to settings.google_cse_id)
        max_results: Max results per query (API allows up to 10 per request)
        rate_limit_delay: Seconds to sleep between requests to avoid 429
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cse_id: Optional[str] = None,
        max_results: int = 10,
        rate_limit_delay: float = 1.0,
    ) -> None:
        self.api_key = api_key or settings.google_api_key
        self.cse_id = cse_id or settings.google_cse_id
        self.max_results = max_results
        self.rate_limit_delay = rate_limit_delay
        self._client = httpx.Client(timeout=15.0)

    @retry(stop=stop_after_attempt(1))
    def search(self, query: str, language: str = "lang_he") -> List[GoogleSearchResult]:
        """
        Execute a Google Custom Search query.

        Args:
            query:    Free-text search query (supports site: operator)
            language: Language restriction for results (default: Hebrew)

        Returns:
            List of GoogleSearchResult objects (up to max_results)

        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx status
        """
        if not self.api_key or not self.cse_id:
            logger.warning(
                "Google API key or CSE ID not configured — returning empty results. "
                "Set GOOGLE_API_KEY and GOOGLE_CSE_ID in .env to enable real search."
            )
            return []

        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": min(self.max_results, 10),
            "lr": language,
        }

        logger.debug(f"Google search query: {query!r}")
        response = self._client.get(GOOGLE_CSE_ENDPOINT, params=params)

        if response.status_code != 200:
            logger.error(
                f"Google API error | status={response.status_code} | "
                f"query={query!r} | response={response.text[:300]}"
            )
        if response.status_code == 429:
            logger.warning("Google API daily quota exhausted (100 queries/day). Try again tomorrow.")
        response.raise_for_status()

        items = response.json().get("items", [])
        results = [
            GoogleSearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
            )
            for item in items
        ]

        logger.info(f"Google search '{query}' → {len(results)} results")
        time.sleep(self.rate_limit_delay)
        return results

    def search_jobs_on_domain(
        self, role: str, domain: str
    ) -> List[GoogleSearchResult]:
        """
        Search for a specific job role on a given ATS domain.
        Uses Google's site: operator to penetrate ATS job boards.

        Example query: "QA Automation Engineer site:comeet.com"

        Args:
            role:   Job title to search for (e.g. "QA Automation Engineer")
            domain: ATS domain (e.g. "comeet.com", "jobs.lever.co")

        Returns:
            List of GoogleSearchResult with job postings
        """
        query = f"{role} site:{domain}"
        return self.search(query, language="")  # No language filter for English domains

    def search_funding_news(
        self, keywords: Optional[List[str]] = None, lang: str = "lang_he"
    ) -> List[GoogleSearchResult]:
        """
        Search for companies that recently raised funding.

        Args:
            keywords: List of funding-related keywords to OR-combine
            lang:     Language filter ("lang_he" for Hebrew news, "" for English)

        Returns:
            List of GoogleSearchResult for funding news
        """
        terms = keywords or settings.funding_keywords_hebrew
        query = " OR ".join(f'"{kw}"' for kw in terms)
        return self.search(query, language=lang)

    def close(self) -> None:
        """Release the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "GoogleSearchTool":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
