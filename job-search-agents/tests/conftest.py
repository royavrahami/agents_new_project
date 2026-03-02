"""
Shared pytest fixtures for all agent and tool tests.
Uses an in-memory SQLite database so tests are fully isolated.
"""

import pytest
from unittest.mock import MagicMock
from uuid import uuid4

# Force in-memory SQLite before any settings are loaded
import os
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("GOOGLE_API_KEY", "")
os.environ.setdefault("GOOGLE_CSE_ID", "")

from core.database import Base, engine, init_db
from core.models import HiddenJob, FundingEvent, PipelineEntry, Contact, OutreachMessage
from config.settings import JobSearchStatus
from tools.llm_tool import LLMTool


# ---------------------------------------------------------------------------
# Database lifecycle
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_database():
    """
    Create all tables before each test and drop them after.
    Ensures complete test isolation with no data leakage between tests.
    """
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


# ---------------------------------------------------------------------------
# Mock LLM — prevents real API calls in tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm() -> LLMTool:
    """
    Returns a LLMTool with the complete() method mocked.
    Default return value is a safe, parseable string.
    Tests that need specific LLM responses should configure mock_llm.complete.return_value.
    """
    llm = MagicMock(spec=LLMTool)
    llm.complete.return_value = "1. First recommendation\n2. Second recommendation\n3. Third recommendation"
    llm.score_relevance.return_value = 0.75
    llm.extract_company_name.return_value = "Acme Corp"
    return llm


# ---------------------------------------------------------------------------
# Domain object factories
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_hidden_job() -> HiddenJob:
    return HiddenJob(
        company_name="Acme Tech",
        role_title="QA Automation Engineer",
        job_url="https://jobs.lever.co/acme/qa-engineer-123",
        source_domain="jobs.lever.co",
        description_snippet="Seeking QA Automation Engineer with Python and Playwright experience.",
        location="Tel Aviv",
        remote=False,
        hot_score=0.82,
        funding_linked=True,
    )


@pytest.fixture
def sample_funding_event() -> FundingEvent:
    return FundingEvent(
        company_name="Acme Tech",
        amount="$10M",
        round_type="Series A",
        source_url="https://calcalist.co.il/article/acme-funding",
        headline="Acme Tech raises $10M Series A",
        relevance_score=0.9,
    )


@pytest.fixture
def sample_pipeline_entry() -> PipelineEntry:
    return PipelineEntry(
        company_name="Acme Tech",
        role_title="QA Automation Engineer",
        job_url="https://jobs.lever.co/acme/qa-engineer-123",
        status=JobSearchStatus.APPLIED,
        hot_score=0.82,
    )


@pytest.fixture
def sample_contact() -> Contact:
    return Contact(
        company_name="Acme Tech",
        full_name="Jane Recruiter",
        title="Technical Recruiter",
        linkedin_url="https://linkedin.com/in/jane-recruiter",
        is_recruiter=True,
    )


@pytest.fixture
def sample_cv_text() -> str:
    return """
    John Doe | QA Automation Engineer
    Tel Aviv, Israel | john.doe@email.com | LinkedIn: /in/johndoe

    SUMMARY
    Senior QA Automation Engineer with 7 years of experience building robust test frameworks
    using Python, Playwright, Selenium, and pytest. Expert in CI/CD integration and API testing.

    EXPERIENCE
    Senior QA Automation Engineer | StartupXYZ | 2021 - Present
    - Built end-to-end test automation framework using Playwright and Python
    - Reduced regression test cycle from 4 hours to 25 minutes
    - Implemented API testing suite with 300+ tests using pytest and requests

    QA Engineer | TechCorp | 2018 - 2021
    - Led QA team of 4 engineers
    - Implemented CI/CD quality gates in Jenkins pipeline

    SKILLS
    Python, Playwright, Selenium, pytest, Jenkins, Docker, API Testing, SQL, Git
    """


@pytest.fixture
def sample_job_description() -> str:
    return """
    QA Automation Engineer

    We are looking for an experienced QA Automation Engineer to join our team.

    Requirements:
    - 5+ years of QA automation experience
    - Strong Python programming skills
    - Experience with Playwright or Selenium
    - Familiarity with CI/CD pipelines (Jenkins, GitHub Actions)
    - API testing experience (REST, GraphQL)
    - Experience with Docker and containerization
    - Knowledge of SQL and database testing

    Nice to have:
    - Experience with performance testing (Locust, k6)
    - Cloud experience (AWS, GCP)
    - Kubernetes knowledge
    """
