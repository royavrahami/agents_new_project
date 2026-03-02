"""
News Scraper Tool.
Fetches and parses funding news from Israeli tech news sources
(Calcalist, Geektime, TechCrunch) without requiring paid API access.

Used by Intel Agent to discover companies that recently raised capital
— a strong signal that hiring will follow within weeks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from core.logger import logger


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class NewsArticle:
    """Lightweight news article extracted from a scrape."""
    title: str
    url: str
    source: str
    snippet: str = ""
    published_at: Optional[datetime] = None
    company_mentions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Source definitions — extend this dict to add more news sources
# ---------------------------------------------------------------------------

NEWS_SOURCES: Dict[str, Dict] = {
    "calcalist": {
        "base_url": "https://www.calcalist.co.il",
        "search_url": "https://www.calcalist.co.il/search?q={}",
        "article_selector": "article",
        "title_selector": "h2, h3",
        "link_selector": "a",
        "snippet_selector": "p",
    },
    "geektime": {
        "base_url": "https://www.geektime.co.il",
        "search_url": "https://www.geektime.co.il/?s={}",
        "article_selector": "article",
        "title_selector": "h2",
        "link_selector": "a",
        "snippet_selector": ".entry-summary p",
    },
    "techcrunch": {
        "base_url": "https://techcrunch.com",
        "search_url": "https://techcrunch.com/search/{}",
        "article_selector": "article",
        "title_selector": "h2",
        "link_selector": "a",
        "snippet_selector": "p",
    },
}

# Patterns that strongly indicate a funding round
FUNDING_PATTERNS = [
    r"giy[ue]s[ah]?\s+hon",          # Hebrew: גיוסה הון
    r"raised?\s+\$[\d\.]+[mMbB]",
    r"series\s+[a-eA-E]",
    r"seed\s+round",
    r"closed\s+round",
    r"השלימה\s+גיוס",
    r"מימון",
    r"השקעה",
]

_FUNDING_RE = re.compile("|".join(FUNDING_PATTERNS), re.IGNORECASE | re.UNICODE)


class NewsScraperTool:
    """
    Scrapes tech news websites for funding announcements.

    Args:
        timeout:    HTTP request timeout in seconds
        user_agent: Browser UA to avoid simple bot blocks
    """

    def __init__(
        self,
        timeout: float = 20.0,
        user_agent: str = (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
    ) -> None:
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    def _fetch(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch a URL and return parsed BeautifulSoup, or None on failure.
        Retries up to 3 times with exponential back-off.
        """
        try:
            resp = self._client.get(url)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except httpx.HTTPError as exc:
            logger.warning(f"HTTP error fetching {url}: {exc}")
            raise  # Triggers tenacity retry

    def search_funding_articles(
        self, keywords: List[str], sources: Optional[List[str]] = None
    ) -> List[NewsArticle]:
        """
        Search multiple news sources for funding-related articles.

        Args:
            keywords: List of search keywords (will try each source with each keyword)
            sources:  Source names to query (subset of NEWS_SOURCES keys); None = all

        Returns:
            Deduplicated list of NewsArticle objects matching funding patterns
        """
        active_sources = {
            name: cfg
            for name, cfg in NEWS_SOURCES.items()
            if sources is None or name in sources
        }

        articles: List[NewsArticle] = []
        seen_urls: set[str] = set()

        for source_name, cfg in active_sources.items():
            for keyword in keywords:
                url = cfg["search_url"].format(keyword.replace(" ", "+"))
                logger.info(f"Scraping {source_name} for keyword '{keyword}'")

                soup = self._fetch(url)
                if soup is None:
                    continue

                for article_tag in soup.select(cfg["article_selector"])[:20]:
                    title_tag = article_tag.select_one(cfg["title_selector"])
                    link_tag = article_tag.select_one(cfg["link_selector"])
                    snippet_tag = article_tag.select_one(cfg["snippet_selector"])

                    if not title_tag or not link_tag:
                        continue

                    title = title_tag.get_text(strip=True)
                    href = link_tag.get("href", "")
                    snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

                    # Resolve relative URLs
                    if href.startswith("/"):
                        href = urljoin(cfg["base_url"], href)

                    # Skip if we've seen this URL
                    if href in seen_urls:
                        continue
                    seen_urls.add(href)

                    # Only include articles that mention funding
                    full_text = f"{title} {snippet}"
                    if not _FUNDING_RE.search(full_text):
                        continue

                    articles.append(
                        NewsArticle(
                            title=title,
                            url=href,
                            source=source_name,
                            snippet=snippet,
                            company_mentions=self._extract_company_mentions(full_text),
                        )
                    )

        logger.info(f"Found {len(articles)} funding articles across {len(active_sources)} sources")
        return articles

    def _extract_company_mentions(self, text: str) -> List[str]:
        """
        Heuristic extraction of company names from article text.
        Looks for capitalized multi-word sequences that likely represent companies.

        This is a lightweight approach — a production system would use NER.

        Args:
            text: Raw article text

        Returns:
            List of potential company names
        """
        # Match sequences of 1-3 capitalized/Hebrew words
        pattern = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2}\b")
        candidates = pattern.findall(text)
        # Filter out generic words
        stopwords = {
            "The", "This", "That", "With", "From", "Series", "Seed",
            "Round", "Israel", "Tech", "Company", "Inc", "Ltd",
        }
        return list({c for c in candidates if c not in stopwords})

    def close(self) -> None:
        """Release the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "NewsScraperTool":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
