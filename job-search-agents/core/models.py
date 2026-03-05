"""
Shared Pydantic domain models used across all agents.
These are the core data contracts — every agent reads and writes these types.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl

from config.settings import JobSearchStatus


# ---------------------------------------------------------------------------
# Intelligence Layer
# ---------------------------------------------------------------------------

class FundingEvent(BaseModel):
    """Represents a company funding event discovered by the Intel Agent."""

    id: UUID = Field(default_factory=uuid4)
    company_name: str
    amount: Optional[str] = None          # e.g. "$5M", "900M ILS"
    round_type: Optional[str] = None       # e.g. "Series A", "Seed"
    source_url: str
    headline: str
    published_at: Optional[datetime] = None
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class HiddenJob(BaseModel):
    """A job opportunity discovered through non-standard channels (hidden market)."""

    id: UUID = Field(default_factory=uuid4)
    company_name: str
    role_title: str
    job_url: str
    source_domain: str                     # e.g. "comeet.com"
    description_snippet: Optional[str] = None
    location: Optional[str] = None
    remote: Optional[bool] = None
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    hot_score: float = Field(default=0.0, ge=0.0, le=1.0)
    funding_linked: bool = Field(default=False)  # True if company recently raised

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ---------------------------------------------------------------------------
# Profile Layer
# ---------------------------------------------------------------------------

class CVAnalysis(BaseModel):
    """Result of ATS analysis on a candidate's CV."""

    ats_score: float = Field(ge=0.0, le=100.0)
    keyword_coverage: float = Field(ge=0.0, le=1.0)
    missing_keywords: List[str] = Field(default_factory=list)
    matched_keywords: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)


class TailoredCV(BaseModel):
    """A CV tailored to a specific job description."""

    job_id: UUID
    original_cv_path: str
    tailored_content: str
    ats_score_before: float
    ats_score_after: float
    changes_summary: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Outreach Layer
# ---------------------------------------------------------------------------

class Contact(BaseModel):
    """A recruiter or hiring manager to reach out to."""

    id: UUID = Field(default_factory=uuid4)
    company_name: str
    full_name: Optional[str] = None
    title: Optional[str] = None
    linkedin_url: Optional[str] = None
    email: Optional[str] = None
    is_hiring_manager: bool = False
    is_recruiter: bool = False
    mutual_connections: List[str] = Field(default_factory=list)


class OutreachMessage(BaseModel):
    """A cold outreach message drafted by the Outreach Agent."""

    id: UUID = Field(default_factory=uuid4)
    contact_id: UUID
    job_id: Optional[UUID] = None
    subject: Optional[str] = None
    body: str
    channel: str = Field(default="linkedin")   # "linkedin" | "email"
    sent_at: Optional[datetime] = None
    responded_at: Optional[datetime] = None
    follow_up_due: Optional[datetime] = None
    status: str = Field(default="DRAFT")       # DRAFT | SENT | RESPONDED | GHOSTED

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ---------------------------------------------------------------------------
# Pipeline / Tracker Layer
# ---------------------------------------------------------------------------

class PipelineEntry(BaseModel):
    """One opportunity in the job search pipeline (CRM record)."""

    id: UUID = Field(default_factory=uuid4)
    company_name: str
    role_title: str
    job_url: Optional[str] = None
    status: JobSearchStatus = Field(default=JobSearchStatus.IDENTIFIED)
    applied_at: Optional[datetime] = None
    last_activity_at: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = None
    rejection_reason: Optional[str] = None
    interview_stage: Optional[int] = None     # 1st, 2nd, 3rd interview
    offer_amount: Optional[str] = None
    hot_score: float = Field(default=0.0, ge=0.0, le=1.0)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ---------------------------------------------------------------------------
# Coach Layer
# ---------------------------------------------------------------------------

class InterviewPrep(BaseModel):
    """Interview preparation package for a specific company + role."""

    pipeline_entry_id: UUID
    company_research: str
    likely_questions: List[str] = Field(default_factory=list)
    star_examples: List[str] = Field(default_factory=list)
    questions_to_ask: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ---------------------------------------------------------------------------
# Orchestrator Layer
# ---------------------------------------------------------------------------

class WeeklyReport(BaseModel):
    """Weekly summary produced by the Orchestrator Agent."""

    week_start: datetime
    week_end: datetime
    total_opportunities_found: int = 0
    total_outreach_sent: int = 0
    total_responses: int = 0
    total_interviews: int = 0
    conversion_rate_apply_to_response: float = 0.0
    conversion_rate_response_to_interview: float = 0.0
    top_opportunities: List[HiddenJob] = Field(default_factory=list)
    bottleneck: Optional[str] = None
    recommended_actions: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class OrchestrationStage(str, Enum):
    """Lifecycle stages for a single orchestrator cycle."""

    DISCOVER = "discover"
    PRIORITIZE = "prioritize"
    TAILOR = "tailor"
    OUTREACH = "outreach"
    TRACK = "track"
    COACH = "coach"
    SUMMARIZE = "summarize"


class StageOutcome(BaseModel):
    """Structured output contract for one orchestration stage."""

    stage: OrchestrationStage
    status: str = Field(default="ok")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    attempts: int = 1
    data: Dict[str, object] = Field(default_factory=dict)
    error: Optional[str] = None

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
