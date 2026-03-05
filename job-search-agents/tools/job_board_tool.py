"""
Job Board Direct Scraper Tool.

Searches Israeli and international job boards directly via HTTP —
no API key or Google Custom Search required.

Used by Intel Agent as the primary job discovery channel
(Google Custom Search is an optional enhancement when available).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, quote_plus

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from core.logger import logger
from core.models import HiddenJob


# ---------------------------------------------------------------------------
# Job board search configurations
# ---------------------------------------------------------------------------

JOB_BOARDS: List[Dict[str, Any]] = [
    # -----------------------------------------------------------------------
    # Israeli generalist job boards (high volume, search works directly)
    # -----------------------------------------------------------------------
    {
        "name": "drushim.co.il",
        "search_url": "https://www.drushim.co.il/jobs/?q={}",
        "base_url": "https://www.drushim.co.il",
        "job_selector": "li.job-item, div.job-item, article.job",
        "title_selector": "h2, h3, .job-title, .position",
        "company_selector": ".company-name, .employer, .company",
        "link_selector": "a",
    },
    {
        "name": "alljobs.co.il",
        "search_url": "https://www.alljobs.co.il/SearchResultsGuest.aspx?q={}",
        "base_url": "https://www.alljobs.co.il",
        "job_selector": "li.job, div.job-item, .job-result",
        "title_selector": "h2, h3, .job-title, .title",
        "company_selector": ".company, .employer-name",
        "link_selector": "a",
    },
    {
        "name": "gotfriends.co.il",
        "search_url": "https://www.gotfriends.co.il/jobs/?q={}",
        "base_url": "https://www.gotfriends.co.il",
        "job_selector": "li.job-item, article, .job-row",
        "title_selector": "h2, h3, .job-name, .position-title",
        "company_selector": ".company-name, .company",
        "link_selector": "a",
    },
    {
        "name": "jobmaster.co.il",
        "search_url": "https://www.jobmaster.co.il/jobs/?q={}",
        "base_url": "https://www.jobmaster.co.il",
        "job_selector": "li.job, div.job-item, article",
        "title_selector": "h2, h3, .job-title",
        "company_selector": ".company-name, .company",
        "link_selector": "a",
    },
    # -----------------------------------------------------------------------
    # Israeli tech-specific job boards
    # -----------------------------------------------------------------------
    {
        "name": "ice.co.il",
        "search_url": "https://www.ice.co.il/jobs/search?q={}",
        "base_url": "https://www.ice.co.il",
        "job_selector": "article, li.job, div.job-item, .job-listing",
        "title_selector": "h2, h3, .job-title, .title",
        "company_selector": ".company, .employer",
        "link_selector": "a",
    },
    {
        "name": "matrix-global.com",
        "search_url": "https://jobs.matrix-global.com/?s={}",
        "base_url": "https://jobs.matrix-global.com",
        "job_selector": "article, li, .job-post",
        "title_selector": "h2, h3, .job-title",
        "company_selector": ".company, .location",
        "link_selector": "a",
    },
    # -----------------------------------------------------------------------
    # ATS platforms with global search (many Israeli startups post here)
    # -----------------------------------------------------------------------
    {
        "name": "comeet.co",
        "search_url": "https://www.comeet.co/jobs/search?q={}&country=IL",
        "base_url": "https://www.comeet.co",
        "job_selector": "li, article, .position-item, .job-item",
        "title_selector": "h2, h3, .position-name, .job-title",
        "company_selector": ".company-name, .company",
        "link_selector": "a",
    },
    {
        "name": "smartrecruiters.com",
        "search_url": "https://jobs.smartrecruiters.com/?q={}&location=Israel",
        "base_url": "https://jobs.smartrecruiters.com",
        "job_selector": "li.job-item, article, .job",
        "title_selector": "h2, h4, .job-title",
        "company_selector": ".company-name, .company",
        "link_selector": "a",
    },
    {
        "name": "pinpointhq.com",
        "search_url": "https://pinpointhq.com/jobs?search={}",
        "base_url": "https://pinpointhq.com",
        "job_selector": "li, article, .job-listing",
        "title_selector": "h2, h3, .job-title",
        "company_selector": ".company, .employer",
        "link_selector": "a",
    },
    {
        "name": "teamtailor.com",
        "search_url": "https://jobs.teamtailor.com/?search={}",
        "base_url": "https://jobs.teamtailor.com",
        "job_selector": "article, li.job",
        "title_selector": "h2, h3, .job-title",
        "company_selector": ".company-name",
        "link_selector": "a",
    },
]

_LOCATION_RE = re.compile(
    r"\b(Tel Aviv|Jerusalem|Haifa|Herzliya|Ra'anana|Petah Tikva|Remote|"
    r"Israel|תל אביב|ירושלים|חיפה|הרצליה|רעננה|פתח תקווה|כפר סבא)\b",
    re.IGNORECASE | re.UNICODE,
)
_REMOTE_RE = re.compile(r"\bremote\b|\bwork.from.home\b|\bwfh\b", re.IGNORECASE)

# Patterns that indicate a URL is a real job posting (not navigation/category)
_JOB_URL_RE = re.compile(
    r"/job[s]?/|/position[s]?/|/career[s]?/[^/]+/[^/]+|/vacancy|"
    r"apply\.|/opening|/role|comeet\.co/jobs/[^/]+/[^/]+/[^/]|"
    r"drushim\.co\.il/jobs/\d|alljobs.*\d{4,}|lever\.co/|greenhouse\.io/",
    re.IGNORECASE,
)

# Job-related keywords — title must contain at least one to be a real job posting
_JOB_TITLE_KEYWORDS = re.compile(
    r"\b(qa|qe|quality|test|automation|engineer|manager|lead|director|"
    r"developer|devops|architect|analyst|head of|vp|team lead|"
    r"מנהל|מפתח|מהנדס|בדיקות|אוטומציה|ראש צוות)\b",
    re.IGNORECASE | re.UNICODE,
)

# Maximum title length — navigation items tend to be long Hebrew phrases
_MAX_TITLE_LEN = 80


class JobBoardTool:
    """
    Scrapes Israeli and international job boards for QA/quality engineering roles.
    Does not require any API key — uses direct HTTP requests.

    Args:
        timeout:    HTTP request timeout in seconds
        user_agent: Browser User-Agent string
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
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "he-IL,he;q=0.9,en-US;q=0.8,en;q=0.7",
            },
            follow_redirects=True,
        )

    _NON_RETRYABLE = frozenset({401, 403, 404, 410})

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
    def _fetch(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch a URL and parse as HTML. Returns None on permanent errors."""
        try:
            resp = self._client.get(url)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if code in self._NON_RETRYABLE:
                logger.debug(f"Permanent HTTP {code} for {url} — skipping")
                return None
            raise
        except httpx.HTTPError as exc:
            logger.debug(f"HTTP error fetching {url}: {exc}")
            raise

    def search_jobs(
        self,
        roles: List[str],
        boards: Optional[List[str]] = None,
        max_per_board: int = 10,
    ) -> List[HiddenJob]:
        """
        Search job boards for the given roles.

        Args:
            roles:         List of job titles to search (e.g. ["QA Manager", "Head of QA"])
            boards:        Subset of board names to query; None = all boards
            max_per_board: Maximum job results to collect per role per board

        Returns:
            Deduplicated list of HiddenJob objects found across all boards
        """
        active_boards = [
            b for b in JOB_BOARDS
            if boards is None or b["name"] in boards
        ]

        jobs: List[HiddenJob] = []
        seen_urls: set[str] = set()

        for role in roles:
            for board in active_boards:
                url = board["search_url"].format(quote_plus(role))
                logger.info(f"Searching {board['name']} for '{role}'")

                try:
                    soup = self._fetch(url)
                except Exception as exc:
                    logger.warning(f"Failed to fetch {board['name']}: {exc}")
                    continue

                if soup is None:
                    continue

                found = self._extract_jobs(soup, board, role, seen_urls, max_per_board)
                jobs.extend(found)
                logger.info(f"  {board['name']}: {len(found)} jobs found for '{role}'")

        logger.info(
            f"Job board search complete: {len(jobs)} total jobs "
            f"across {len(active_boards)} boards"
        )
        return jobs

    def _extract_jobs(
        self,
        soup: BeautifulSoup,
        board: Dict[str, Any],
        role: str,
        seen_urls: set[str],
        max_results: int,
    ) -> List[HiddenJob]:
        """
        Extract job listings from a parsed HTML page.

        Tries multiple selectors and falls back to generic `<a>` tag
        extraction when structured selectors yield nothing.

        Args:
            soup:        Parsed HTML
            board:       Board config dict
            role:        Search role (used as fallback title)
            seen_urls:   Set of already-seen URLs to deduplicate
            max_results: Maximum jobs to return

        Returns:
            List of HiddenJob objects
        """
        jobs: List[HiddenJob] = []

        # Try structured selectors first
        items = soup.select(board["job_selector"])[:max_results * 2]

        # Fallback: look for any anchor tags that look like job links
        if not items:
            items = [
                a.parent for a in soup.find_all("a", href=True)
                if any(
                    kw in (a.get("href", "") + a.get_text()).lower()
                    for kw in ["job", "position", "career", "משרה"]
                )
            ][:max_results * 2]

        for item in items:
            if len(jobs) >= max_results:
                break

            title_tag = item.select_one(board["title_selector"])
            link_tag = item.select_one(board["link_selector"])
            company_tag = item.select_one(board["company_selector"])

            if not link_tag:
                continue

            href = link_tag.get("href", "")
            if not href or href == "#":
                continue
            if not href.startswith("http"):
                href = urljoin(board["base_url"], href)
            if href in seen_urls:
                continue

            title = (
                title_tag.get_text(strip=True)
                if title_tag
                else link_tag.get_text(strip=True) or role
            )

            # --- Quality filters: skip navigation, categories, and non-job links ---

            # Skip overly long titles (navigation/promo text)
            if len(title) > _MAX_TITLE_LEN:
                continue

            # Skip if title has no job-related keywords
            if not _JOB_TITLE_KEYWORDS.search(title):
                continue

            # Skip if URL does not look like a specific job posting
            if not _JOB_URL_RE.search(href):
                continue

            seen_urls.add(href)

            # Extract company name — try multiple selectors
            company = "Unknown"
            if company_tag:
                company = company_tag.get_text(strip=True)
            else:
                # Fallback: look for company-like text in sibling/parent elements
                for selector in [".company", ".employer", ".company-name",
                                  "[class*='company']", "[class*='employer']"]:
                    found = item.select_one(selector)
                    if found:
                        company = found.get_text(strip=True)
                        break

            full_text = item.get_text(" ", strip=True)
            location = self._extract_location(full_text)
            is_remote = bool(_REMOTE_RE.search(full_text))

            jobs.append(
                HiddenJob(
                    company_name=company or "Unknown",
                    role_title=title,
                    job_url=href,
                    source_domain=board["name"],
                    description_snippet=full_text[:400],
                    location=location,
                    remote=is_remote,
                )
            )

        return jobs

    @staticmethod
    def _extract_location(text: str) -> Optional[str]:
        """Extract first city/location mention from text."""
        match = _LOCATION_RE.search(text)
        return match.group(0) if match else None

    def close(self) -> None:
        """Release the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "JobBoardTool":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
