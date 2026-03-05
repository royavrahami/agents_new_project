"""
Database engine, session factory, and base model for SQLAlchemy ORM.
All agents use `get_db()` as a context manager to obtain a session.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config.settings import settings
from core.logger import logger


# ---------------------------------------------------------------------------
# Ensure the data directory exists before SQLite tries to create the file
# ---------------------------------------------------------------------------

if "sqlite" in settings.database_url:
    db_path = settings.database_url.replace("sqlite:///", "")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Engine & Session factory
# ---------------------------------------------------------------------------

engine = create_engine(
    settings.database_url,
    # SQLite-specific: allow usage from multiple threads
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ---------------------------------------------------------------------------
# ORM Tables
# ---------------------------------------------------------------------------

class FundingEventORM(Base):
    __tablename__ = "funding_events"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    company_name = Column(String, nullable=False, index=True)
    amount = Column(String)
    round_type = Column(String)
    source_url = Column(Text)
    headline = Column(Text)
    published_at = Column(DateTime)
    discovered_at = Column(DateTime, default=datetime.utcnow)
    relevance_score = Column(Float, default=0.0)


class HiddenJobORM(Base):
    __tablename__ = "hidden_jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    company_name = Column(String, nullable=False, index=True)
    role_title = Column(String, nullable=False)
    job_url = Column(Text, unique=True)
    source_domain = Column(String)
    description_snippet = Column(Text)
    location = Column(String)
    remote = Column(Boolean)
    discovered_at = Column(DateTime, default=datetime.utcnow)
    hot_score = Column(Float, default=0.0)
    funding_linked = Column(Boolean, default=False)


class PipelineEntryORM(Base):
    __tablename__ = "pipeline"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    company_name = Column(String, nullable=False)
    role_title = Column(String, nullable=False)
    job_url = Column(Text)
    status = Column(String, default="IDENTIFIED")
    applied_at = Column(DateTime)
    last_activity_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)
    rejection_reason = Column(Text)
    interview_stage = Column(Integer)
    offer_amount = Column(String)
    hot_score = Column(Float, default=0.0)


class OutreachMessageORM(Base):
    __tablename__ = "outreach_messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    contact_name = Column(String)
    contact_company = Column(String)
    contact_linkedin = Column(String)
    contact_email = Column(String)
    job_id = Column(String)
    subject = Column(String)
    body = Column(Text)
    channel = Column(String, default="linkedin")
    sent_at = Column(DateTime)
    responded_at = Column(DateTime)
    follow_up_due = Column(DateTime)
    status = Column(String, default="DRAFT")


class OrchestratorRunORM(Base):
    """Execution log for orchestrator runs used for idempotency and observability."""

    __tablename__ = "orchestrator_runs"
    __table_args__ = (
        UniqueConstraint("run_key", name="uq_orchestrator_runs_run_key"),
    )

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    run_key = Column(String, nullable=False, index=True)
    cycle_type = Column(String, nullable=False, default="daily")
    correlation_id = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, default="running")
    details = Column(Text)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables if they do not exist. Called once at startup."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized — all tables created/verified.")


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Context manager that yields a SQLAlchemy Session.
    Commits on success, rolls back on any exception, and always closes.

    Usage:
        with get_db() as db:
            db.add(some_orm_object)
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
