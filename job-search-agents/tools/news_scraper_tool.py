"""
News Scraper Tool.
Fetches funding news via RSS feeds from major tech/VC news sources.

RSS-based approach is preferred over HTML scraping because:
- XML is machine-readable by design (no JavaScript rendering issues)
- No rate limits or bot-blocking
- Always returns clean, structured data
- Significantly faster (one request per feed)

Used by Intel Agent to discover companies that recently raised capital
— a strong signal that hiring will follow within weeks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional

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
# RSS feed sources — edit this dict to add more feeds
# ---------------------------------------------------------------------------

RSS_SOURCES: Dict[str, str] = {
    # TechCrunch VC & startup funding category
    "techcrunch_vc": "https://techcrunch.com/category/venture/feed/",
    # TechCrunch startups
    "techcrunch_startups": "https://techcrunch.com/category/startups/feed/",
    # VentureBeat business / enterprise tech
    "venturebeat": "https://venturebeat.com/category/business/feed/",
    # Crunchbase News — dedicated VC/funding news
    "crunchbase_news": "https://news.crunchbase.com/feed/",
}

# Keep NEWS_SOURCES as an alias (used in tests and settings)
NEWS_SOURCES: Dict[str, Dict] = {
    name: {"rss_url": url} for name, url in RSS_SOURCES.items()
}

# English-only patterns that strongly indicate a funding round
FUNDING_PATTERNS = [
    r"raises?\s+\$[\d,\.]+\s*[mMbBkK]",    # raises / raised $5M / $200K
    r"raises?\s+[\d,\.]+\s*million",
    r"series\s+[a-eA-E]\b",                  # Series A / Series B
    r"seed\s+(round|funding)",
    r"closed?\s+(?:a\s+)?\$[\d,\.]+",        # closed a $10M round
    r"\bpre-?seed\b",
    r"growth\s+(equity|round|capital)",
    r"bridge\s+(round|financing)",
    r"\bfunding\b.{0,40}\bmillion\b",
    r"\bmillion\b.{0,40}\bfunding\b",
    r"venture\s+capital\s+funding",
    r"investment\s+round",
    r"led\s+by\s+[A-Z][a-z]+\s+(?:Capital|Ventures|Partners|Fund)",
]

_FUNDING_RE = re.compile("|".join(FUNDING_PATTERNS), re.IGNORECASE)


class NewsScraperTool:
    """
    Fetches and filters tech funding news using RSS feeds.

    RSS feeds are XML-based, publicly accessible, and return structured
    article data without JavaScript rendering, making them far more
    reliable than HTML scraping for production use.

    Args:
        timeout:    HTTP request timeout in seconds
        user_agent: Browser UA string sent with every request
    """

    def __init__(
        self,
        timeout: float = 10.0,
        user_agent: str = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    ) -> None:
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )

    # 410 Gone = resource permanently removed, never retry
    _NON_RETRYABLE_HTTP_CODES = frozenset({401, 403, 404, 410})

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    def _fetch(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch a URL and return a BeautifulSoup object, or None on permanent failure.
        Retries up to 3 times on transient errors (5xx, timeouts).
        Returns None immediately on permanent errors (401, 403, 404).
        """
        try:
            resp = self._client.get(url)
            resp.raise_for_status()
            # Use lxml-xml parser for RSS/Atom feeds, lxml for HTML fallback
            content_type = resp.headers.get("content-type", "")
            parser = "lxml-xml" if ("xml" in content_type or "rss" in content_type) else "lxml"
            return BeautifulSoup(resp.content, parser)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in self._NON_RETRYABLE_HTTP_CODES:
                logger.warning(
                    f"Permanent HTTP {exc.response.status_code} for {url} — skipping (no retry)"
                )
                return None
            logger.warning(f"HTTP error fetching {url}: {exc}")
            raise
        except httpx.HTTPError as exc:
            logger.warning(f"HTTP error fetching {url}: {exc}")
            raise

    def _parse_rss_feed(self, source_name: str, rss_url: str) -> List[NewsArticle]:
        """
        Fetch and parse a single RSS feed, returning funding-related articles.

        Args:
            source_name: Human-readable source label used in NewsArticle.source
            rss_url:     Full RSS/Atom feed URL

        Returns:
            List of NewsArticle objects that mention a funding event
        """
        logger.info(f"Fetching RSS feed: {source_name}")
        soup = self._fetch(rss_url)
        if soup is None:
            return []

        articles: List[NewsArticle] = []
        # RSS uses <item>, Atom uses <entry> — handle both
        items = soup.find_all("item") or soup.find_all("entry")

        for item in items[:30]:  # Process up to 30 items per feed
            title_tag = item.find("title")
            link_tag = item.find("link")
            desc_tag = (
                item.find("description")
                or item.find("summary")
                or item.find("content")
            )
            pub_tag = item.find("pubDate") or item.find("published") or item.find("updated")

            if not title_tag:
                continue

            title = title_tag.get_text(strip=True)

            # Atom feeds encode the URL as <link href="..."/> text node or attribute
            if link_tag:
                href = link_tag.get("href") or link_tag.get_text(strip=True)
            else:
                href = ""

            # Strip HTML tags from description/summary
            raw_desc = desc_tag.get_text(strip=True) if desc_tag else ""
            # Further clean encoded HTML that may appear in RSS descriptions
            snippet = re.sub(r"<[^>]+>", " ", raw_desc)[:500].strip()

            published_at: Optional[datetime] = None
            if pub_tag:
                try:
                    published_at = parsedate_to_datetime(pub_tag.get_text(strip=True))
                except Exception:
                    pass

            full_text = f"{title} {snippet}"
            if not _FUNDING_RE.search(full_text):
                continue

            articles.append(
                NewsArticle(
                    title=title,
                    url=href,
                    source=source_name,
                    snippet=snippet,
                    published_at=published_at,
                    company_mentions=self._extract_company_mentions(full_text),
                )
            )

        logger.info(f"  {source_name}: {len(articles)} funding articles found")
        return articles

    def search_funding_articles(
        self, keywords: List[str], sources: Optional[List[str]] = None
    ) -> List[NewsArticle]:
        """
        Fetch RSS feeds and return articles that match funding patterns.

        The keywords parameter is used to additionally filter articles
        (beyond the built-in FUNDING_PATTERNS) so callers can narrow
        results to specific funding types (e.g. "Series A").

        Args:
            keywords: Optional extra keyword filters applied on top of FUNDING_PATTERNS
            sources:  RSS source names to query (subset of RSS_SOURCES keys); None = all

        Returns:
            Deduplicated list of NewsArticle objects matching funding patterns
        """
        active_sources = {
            name: url
            for name, url in RSS_SOURCES.items()
            if sources is None or name in sources
        }

        if not active_sources:
            logger.warning(
                f"No matching RSS sources found for filter {sources}. "
                f"Available sources: {list(RSS_SOURCES.keys())}"
            )
            return []

        articles: List[NewsArticle] = []
        seen_urls: set[str] = set()

        for source_name, rss_url in active_sources.items():
            try:
                feed_articles = self._parse_rss_feed(source_name, rss_url)
            except Exception as exc:
                logger.warning(f"Failed to fetch RSS feed {source_name}: {exc}")
                continue

            for article in feed_articles:
                if article.url in seen_urls:
                    continue
                seen_urls.add(article.url)

                # Apply optional keyword filter (any keyword match = include)
                if keywords:
                    text_lower = f"{article.title} {article.snippet}".lower()
                    if not any(kw.lower() in text_lower for kw in keywords):
                        continue

                articles.append(article)

        logger.info(
            f"Found {len(articles)} funding articles across {len(active_sources)} RSS sources"
        )
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
        pattern = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2}\b")
        candidates = pattern.findall(text)
        stopwords = {
            "The", "This", "That", "With", "From", "Series", "Seed",
            "Round", "Israel", "Tech", "Company", "Inc", "Ltd", "New",
            "Venture", "Capital", "Fund", "Partners", "Management",
        }
        return list({c for c in candidates if c not in stopwords})

    def close(self) -> None:
        """Release the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "NewsScraperTool":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
