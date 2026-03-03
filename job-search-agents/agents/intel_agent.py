"""
Intel Agent — Hidden Job Market Scout.

Responsibilities:
1. Scan news sources for companies that recently raised funding (strong hiring signal)
2. Search hidden job boards via Google site: operator
3. Score each opportunity (hot_score) and persist to database
4. Flag jobs at recently-funded companies with `funding_linked=True`

This is the highest-ROI agent: it surfaces opportunities BEFORE they are posted
publicly, eliminating competition from the standard application funnel.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from config.settings import settings
from core.database import FundingEventORM, HiddenJobORM, get_db
from core.logger import logger
from core.models import FundingEvent, HiddenJob
from tools.google_search_tool import GoogleSearchResult, GoogleSearchTool
from tools.llm_tool import LLMTool
from tools.news_scraper_tool import NewsArticle, NewsScraperTool

from .base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Scoring weights — tune these based on signal quality
# ---------------------------------------------------------------------------
_SCORE_WEIGHTS = {
    "funding_linked": 0.40,   # Company recently raised → high hiring probability
    "role_keyword_match": 0.35,  # Job title matches target keywords
    "recency": 0.15,          # Discovered within last 24 hours
    "remote": 0.10,           # Remote position bonus
}


class IntelAgent(BaseAgent):
    """
    Scans the hidden job market and funding news to surface
    high-probability opportunities before they're publicly posted.

    Args:
        google_tool:  GoogleSearchTool instance (injected for testability)
        news_tool:    NewsScraperTool instance (injected for testability)
        llm_tool:     LLMTool instance (injected for testability)
        target_roles: List of job titles to search for
        target_keywords: Keywords used to score relevance
    """

    def __init__(
        self,
        google_tool: Optional[GoogleSearchTool] = None,
        news_tool: Optional[NewsScraperTool] = None,
        llm_tool: Optional[LLMTool] = None,
        target_roles: Optional[List[str]] = None,
        target_keywords: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name="IntelAgent")

        self._google = google_tool or GoogleSearchTool()
        self._news = news_tool or NewsScraperTool()
        self._llm = llm_tool or LLMTool(
            system_prompt=(
                "You are an expert recruiter helping identify job opportunities "
                "in the Israeli tech market."
            )
        )

        self._target_roles = target_roles or [
            settings.candidate_role,          # "QA Manager"
            "Head of QA",
            "Quality Engineering Manager",
            "QA Team Lead",
            "VP Quality",
        ]
        self._target_keywords = target_keywords or settings.target_keywords

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the full intel scan cycle:
        1. Discover funded companies
        2. Search hidden job boards
        3. Score and persist all findings

        Returns:
            {
                "status": "ok",
                "funding_events": int,
                "hidden_jobs": int,
                "hot_jobs": List[HiddenJob],
                "summary": str
            }
        """
        self.logger.info("Starting Intel scan cycle")

        # Step 1 — Funding news
        funded_companies = self._scan_funding_news()
        funded_company_names = {c.company_name.lower() for c in funded_companies}

        # Step 2 — Hidden job board search
        raw_jobs = self._search_hidden_job_boards()

        # Step 3 — Score and enrich
        scored_jobs = self._score_and_enrich_jobs(raw_jobs, funded_company_names)

        # Step 4 — Persist everything
        self._persist_funding_events(funded_companies)
        self._persist_hidden_jobs(scored_jobs)

        # Step 5 — Return summary
        hot_jobs = [j for j in scored_jobs if j.hot_score >= 0.6]
        hot_jobs.sort(key=lambda j: j.hot_score, reverse=True)

        summary = (
            f"Intel scan complete: "
            f"{len(funded_companies)} funding events, "
            f"{len(scored_jobs)} hidden jobs found, "
            f"{len(hot_jobs)} hot opportunities (score ≥ 0.6)"
        )
        self.logger.success(summary)

        return {
            "status": "ok",
            "funding_events": len(funded_companies),
            "hidden_jobs": len(scored_jobs),
            "hot_jobs": hot_jobs[:10],
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Step 1: Funding news scan
    # ------------------------------------------------------------------

    def _scan_funding_news(self) -> List[FundingEvent]:
        """
        Query news sources for companies that recently raised capital.

        Returns:
            List of FundingEvent domain objects
        """
        self.logger.info("Scanning funding news sources")
        articles: List[NewsArticle] = []

        # Hebrew sources
        try:
            articles += self._news.search_funding_articles(
                keywords=settings.funding_keywords_hebrew,
                sources=["calcalist", "geektime"],
            )
        except Exception as exc:
            self.logger.warning(f"News scrape failed (Hebrew sources): {exc}")

        # English sources via Google
        try:
            google_results = self._google.search_funding_news(
                keywords=settings.funding_keywords_english, lang=""
            )
            articles += self._convert_google_to_articles(google_results)
        except Exception as exc:
            self.logger.warning(f"Google funding search failed: {exc}")

        events = [self._article_to_funding_event(a) for a in articles]
        self.logger.info(f"Funding scan: {len(events)} events discovered")
        return events

    def _convert_google_to_articles(
        self, results: List[GoogleSearchResult]
    ) -> List[NewsArticle]:
        from tools.news_scraper_tool import NewsArticle
        return [
            NewsArticle(
                title=r.title,
                url=r.url,
                source="google",
                snippet=r.snippet,
            )
            for r in results
        ]

    def _article_to_funding_event(self, article: NewsArticle) -> FundingEvent:
        """
        Convert a news article into a FundingEvent domain model.
        Uses LLM to extract company name and round info if not obvious.
        """
        # Try to extract amount from title/snippet
        amount = self._extract_amount(f"{article.title} {article.snippet}")
        round_type = self._extract_round_type(f"{article.title} {article.snippet}")

        # Use LLM for company name extraction when heuristics fail
        company_name = (
            article.company_mentions[0]
            if article.company_mentions
            else (self._llm.extract_company_name(article.title) or "Unknown")
        )

        return FundingEvent(
            company_name=company_name,
            amount=amount,
            round_type=round_type,
            source_url=article.url,
            headline=article.title,
            published_at=article.published_at,
            relevance_score=0.8,  # All funding news is relevant
        )

    # ------------------------------------------------------------------
    # Step 2: Hidden job board search
    # ------------------------------------------------------------------

    def _search_hidden_job_boards(self) -> List[HiddenJob]:
        """
        Use Google site: operator to find job postings on ATS platforms
        that may not be indexed or visible through standard channels.

        Returns:
            List of HiddenJob domain objects
        """
        jobs: List[HiddenJob] = []
        seen_urls: set[str] = set()

        for role in self._target_roles:
            for domain in settings.job_board_domains:
                self.logger.info(f"Searching '{role}' on {domain}")
                try:
                    results = self._google.search_jobs_on_domain(role=role, domain=domain)
                except Exception as exc:
                    self.logger.warning(f"Job search failed for {domain}: {exc}")
                    continue

                for result in results:
                    if result.url in seen_urls:
                        continue
                    seen_urls.add(result.url)

                    job = HiddenJob(
                        company_name=self._extract_company_from_job(result),
                        role_title=self._extract_role_title(result.title) or role,
                        job_url=result.url,
                        source_domain=domain,
                        description_snippet=result.snippet,
                        location=self._extract_location(result.snippet),
                        remote=self._is_remote(result.snippet),
                    )
                    jobs.append(job)

        self.logger.info(f"Job board search: {len(jobs)} raw jobs discovered")
        return jobs

    def _extract_company_from_job(self, result: GoogleSearchResult) -> str:
        """
        Extract company name from Google search result metadata.
        Google often includes 'at CompanyName' in job titles.
        """
        match = re.search(r"\bat\s+([A-Z][A-Za-z0-9\s\-&\.]+)", result.title)
        if match:
            return match.group(1).strip()
        # Fall back to domain-derived or LLM extraction
        return self._llm.extract_company_name(result.title) or "Unknown Company"

    def _extract_role_title(self, title: str) -> Optional[str]:
        """Clean up job title from Google result title."""
        # Remove common suffixes like "| Careers", "- Comeet", etc.
        cleaned = re.sub(r"\s*[\|\-–]\s*.+$", "", title).strip()
        return cleaned if cleaned else None

    def _extract_location(self, text: str) -> Optional[str]:
        """Heuristic location extraction from job snippet."""
        patterns = [
            r"\b(Tel Aviv|Jerusalem|Haifa|Be\'er Sheva|Herzliya|Ra\'anana|Petah Tikva|Remote)\b",
            r"\b(Israel|תל אביב|ירושלים|חיפה)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _is_remote(self, text: str) -> bool:
        """Check if a job mentions remote work."""
        return bool(re.search(r"\bremote\b|\bwork from home\b|\bwfh\b", text, re.IGNORECASE))

    # ------------------------------------------------------------------
    # Step 3: Scoring
    # ------------------------------------------------------------------

    def _score_and_enrich_jobs(
        self, jobs: List[HiddenJob], funded_company_names: set[str]
    ) -> List[HiddenJob]:
        """
        Calculate a hot_score for each job using weighted signals.

        Scoring factors:
        - funding_linked: Company recently raised capital
        - role_keyword_match: Title matches target keywords
        - recency: Just discovered
        - remote: Remote work available

        Args:
            jobs: Raw list of HiddenJob objects
            funded_company_names: Set of lowercase company names that recently raised

        Returns:
            Same list with hot_score and funding_linked fields populated
        """
        scored = []
        for job in jobs:
            score = 0.0

            # Signal 1: Company recently raised funding
            is_funded = job.company_name.lower() in funded_company_names
            if is_funded:
                score += _SCORE_WEIGHTS["funding_linked"]
                job.funding_linked = True

            # Signal 2: Role keyword match
            title_lower = job.role_title.lower()
            keyword_hits = sum(
                1 for kw in self._target_keywords if kw.lower() in title_lower
            )
            if keyword_hits > 0:
                ratio = min(keyword_hits / max(len(self._target_keywords), 1), 1.0)
                score += _SCORE_WEIGHTS["role_keyword_match"] * ratio

            # Signal 3: Recency (always fresh in a single run)
            score += _SCORE_WEIGHTS["recency"]

            # Signal 4: Remote bonus
            if job.remote:
                score += _SCORE_WEIGHTS["remote"]

            job.hot_score = round(min(score, 1.0), 3)
            scored.append(job)

        return scored

    # ------------------------------------------------------------------
    # Step 4: Persistence
    # ------------------------------------------------------------------

    def _persist_funding_events(self, events: List[FundingEvent]) -> None:
        """Save funding events to the database, skipping duplicates."""
        if not events:
            return
        with get_db() as db:
            for event in events:
                existing = (
                    db.query(FundingEventORM)
                    .filter_by(source_url=event.source_url)
                    .first()
                )
                if existing:
                    continue
                orm = FundingEventORM(
                    id=str(event.id),
                    company_name=event.company_name,
                    amount=event.amount,
                    round_type=event.round_type,
                    source_url=event.source_url,
                    headline=event.headline,
                    published_at=event.published_at,
                    relevance_score=event.relevance_score,
                )
                db.add(orm)
        self.logger.info(f"Persisted {len(events)} funding events")

    def _persist_hidden_jobs(self, jobs: List[HiddenJob]) -> None:
        """Save hidden jobs to the database, skipping duplicates by URL."""
        if not jobs:
            return
        new_count = 0
        with get_db() as db:
            for job in jobs:
                existing = (
                    db.query(HiddenJobORM).filter_by(job_url=job.job_url).first()
                )
                if existing:
                    continue
                orm = HiddenJobORM(
                    id=str(job.id),
                    company_name=job.company_name,
                    role_title=job.role_title,
                    job_url=job.job_url,
                    source_domain=job.source_domain,
                    description_snippet=job.description_snippet,
                    location=job.location,
                    remote=job.remote,
                    hot_score=job.hot_score,
                    funding_linked=job.funding_linked,
                )
                db.add(orm)
                new_count += 1
        self.logger.info(f"Persisted {new_count} new hidden jobs (skipped duplicates)")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_amount(text: str) -> Optional[str]:
        """Extract funding amount from text (e.g. '$5M', '900M ILS')."""
        match = re.search(
            r"\$[\d,\.]+\s*[mMbBkK]|[\d,\.]+\s*(?:million|billion|מיליון|מיליארד)",
            text,
            re.IGNORECASE | re.UNICODE,
        )
        return match.group(0).strip() if match else None

    @staticmethod
    def _extract_round_type(text: str) -> Optional[str]:
        """Extract funding round type from text."""
        match = re.search(
            r"\b(seed|series\s+[a-eA-E]|pre-seed|growth|bridge)\b",
            text,
            re.IGNORECASE,
        )
        return match.group(0).strip() if match else None

    def get_recent_hot_jobs(self, limit: int = 10) -> List[HiddenJob]:
        """
        Retrieve the most recent high-scoring jobs from the database.
        Useful for the Orchestrator's daily briefing.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of HiddenJob sorted by hot_score descending
        """
        with get_db() as db:
            rows = (
                db.query(HiddenJobORM)
                .order_by(HiddenJobORM.hot_score.desc(), HiddenJobORM.discovered_at.desc())
                .limit(limit)
                .all()
            )
        return [
            HiddenJob(
                company_name=r.company_name,
                role_title=r.role_title,
                job_url=r.job_url or "",
                source_domain=r.source_domain or "",
                description_snippet=r.description_snippet,
                location=r.location,
                remote=r.remote,
                hot_score=r.hot_score or 0.0,
                funding_linked=r.funding_linked or False,
            )
            for r in rows
        ]
