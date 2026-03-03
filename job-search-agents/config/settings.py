"""
Central configuration module.
All environment variables and application settings are defined here.
Uses pydantic-settings for type-safe, validated config loading from .env.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class JobSearchStatus(str, Enum):
    IDENTIFIED = "IDENTIFIED"
    APPLIED = "APPLIED"
    RESPONDED = "RESPONDED"
    INTERVIEWING = "INTERVIEWING"
    OFFER = "OFFER"
    REJECTED = "REJECTED"
    GHOSTED = "GHOSTED"


# ---------------------------------------------------------------------------
# Main settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables or .env file.
    All sensitive values (API keys) must be set via environment — never hardcoded.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM ---
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model to use")
    openai_temperature: float = Field(default=0.3, ge=0.0, le=2.0)

    # --- Database ---
    database_url: str = Field(
        default="sqlite:///./data/job_search.db",
        description="SQLAlchemy database URL",
    )

    # --- Redis / Celery ---
    redis_url: str = Field(default="redis://localhost:6379/0")

    # --- Telegram ---
    telegram_bot_token: str = Field(default="", description="Telegram bot token")
    telegram_chat_id: str = Field(default="", description="Telegram chat ID")

    # --- Google Search ---
    google_api_key: str = Field(default="", description="Google Custom Search API key")
    google_cse_id: str = Field(default="", description="Google Custom Search Engine ID")

    # --- CV Parser (Affinda/Textkernel) ---
    affinda_api_key: str = Field(default="", description="Affinda CV parsing API key")
    textkernel_api_key: str = Field(default="", description="Textkernel CV parsing API key")

    # --- LinkedIn ---
    linkedin_access_token: str = Field(default="", description="LinkedIn API access token")
    linkedin_refresh_token: str = Field(default="", description="LinkedIn API refresh token")

    # --- Email Discovery (Hunter.io / Apollo.io) ---
    hunter_api_key: str = Field(default="", description="Hunter.io email discovery API key")
    apollo_api_key: str = Field(default="", description="Apollo.io contact discovery API key")

    # --- Comeet ---
    comeet_api_key: str = Field(default="", description="Comeet job board API key")

    # --- Gmail ---
    gmail_access_token: str = Field(default="", description="Gmail API access token")
    gmail_refresh_token: str = Field(default="", description="Gmail API refresh token")

    # --- Calendly ---
    calendly_api_key: str = Field(default="", description="Calendly API key")

    # --- Caching & Rate Limiting ---
    contact_cache_ttl_days: int = Field(default=30, ge=1, le=365, description="Contact discovery cache TTL in days")
    company_research_cache_ttl_days: int = Field(default=30, ge=1, le=365, description="Company research cache TTL in days")

    # --- Candidate profile ---
    candidate_name: str = Field(default="Roy Avrahami")
    candidate_role: str = Field(default="QA Manager")
    candidate_location: str = Field(default="Israel")
    candidate_experience_years: int = Field(default=14, ge=0)
    candidate_summary: str = Field(
        default=(
            "QA Manager with 14+ years of experience leading QA strategy, quality architecture, "
            "and cross-functional governance across Deep-Tech, FinTech, and Enterprise environments. "
            "Specialize in building QA departments from scratch, CI/CD quality gates, release governance, "
            "KPI-driven quality operating models, and AI Agent Orchestration. "
            "Technical stack: Python, Pytest, Playwright, Selenium, Docker, GitHub Actions, Jira/Xray, Datadog, Kubernetes."
        )
    )
    target_companies: List[str] = Field(default_factory=list)
    target_keywords: List[str] = Field(
        default_factory=lambda: [
            "QA Manager",
            "Quality Engineering Manager",
            "Head of QA",
            "VP Quality",
            "QA Lead",
            "Quality Assurance Manager",
            "Test Manager",
            "SDET Manager",
        ]
    )

    # --- Intel Agent ---
    funding_scan_interval_hours: int = Field(default=24, ge=1)
    job_board_domains: List[str] = Field(
        default_factory=lambda: [
            "comeet.com",
            "jobs.lever.co",
            "greenhouse.io",
            "smartrecruiters.com",
            "workable.com",
        ]
    )
    funding_keywords_hebrew: List[str] = Field(
        default_factory=lambda: [
            "גייסה הון",
            "השלימה גיוס",
            "גיוס הון",
            "השקעה",
            "מימון",
        ]
    )
    funding_keywords_english: List[str] = Field(
        default_factory=lambda: [
            "raised funding",
            "series A",
            "series B",
            "seed round",
            "closed round",
            "investment round",
        ]
    )

    # --- Outreach Agent ---
    outreach_daily_limit: int = Field(default=15, ge=1, le=50)
    follow_up_days: int = Field(default=5, ge=1)

    # --- Logging ---
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_file: str = Field(default="logs/job_search_agents.log")

    # --- Paths ---
    data_dir: Path = Field(default=Path("data"))
    raw_dir: Path = Field(default=Path("data/raw"))
    processed_dir: Path = Field(default=Path("data/processed"))

    @field_validator("target_companies", "target_keywords", mode="before")
    @classmethod
    def parse_list_from_string(cls, v: object) -> object:
        """
        Accept both formats from .env:
          - Comma-separated: QA Manager,Head of QA,VP Quality
          - JSON array:      ["QA Manager","Head of QA","VP Quality"]
        pydantic-settings v2 tries JSON first; this validator handles both safely.
        """
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            if v.startswith("["):
                import json as _json
                try:
                    return _json.loads(v)
                except Exception:
                    pass
            return [item.strip() for item in v.split(",") if item.strip()]
        return v or []


# Singleton instance — import this everywhere
settings = Settings()
