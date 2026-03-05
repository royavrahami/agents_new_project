"""
Scheduler service for daily and weekly orchestrator cycles.

This module provides:
- Cron-based scheduling with APScheduler
- Run idempotency keys to prevent duplicate execution windows
- Retry with exponential backoff for transient failures
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Callable, Optional
from uuid import uuid4

from apscheduler.schedulers.blocking import BlockingScheduler
from sqlalchemy.exc import IntegrityError

from agents.orchestrator_agent import OrchestratorAgent
from config.settings import settings
from core.database import OrchestratorRunORM, get_db, init_db
from core.logger import logger


def _claim_run_window(cycle_type: str, window_start: datetime, correlation_id: str) -> bool:
    """
    Claim a logical run window for idempotency.

    Returns False if this run window was already claimed by another execution.
    """
    run_key = f"{cycle_type}:{window_start.strftime('%Y-%m-%d-%H-%M')}"
    try:
        with get_db() as db:
            db.add(
                OrchestratorRunORM(
                    run_key=run_key,
                    cycle_type=cycle_type,
                    correlation_id=correlation_id,
                    status="running",
                    started_at=datetime.utcnow(),
                )
            )
        logger.info(f"Claimed run window: {run_key}")
        return True
    except IntegrityError:
        logger.warning(f"Run window already claimed, skipping: {run_key}")
        return False


def _complete_run(correlation_id: str, status: str, details: str = "") -> None:
    """Finalize the orchestration run log entry."""
    with get_db() as db:
        run = (
            db.query(OrchestratorRunORM)
            .filter_by(correlation_id=correlation_id)
            .order_by(OrchestratorRunORM.started_at.desc())
            .first()
        )
        if not run:
            return
        run.status = status
        run.details = details[:4000]
        run.completed_at = datetime.utcnow()


def _run_with_retries(
    run_callable: Callable[[], None],
    correlation_id: str,
    max_attempts: int = 3,
) -> None:
    """Execute a callable with retry and exponential backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            run_callable()
            _complete_run(correlation_id=correlation_id, status="ok")
            return
        except Exception as exc:
            wait_seconds = min(2 ** attempt, 30)
            logger.warning(
                f"Scheduled run attempt {attempt}/{max_attempts} failed: {exc}. "
                f"Retrying in {wait_seconds}s"
            )
            if attempt == max_attempts:
                _complete_run(correlation_id=correlation_id, status="error", details=str(exc))
                raise
            time.sleep(wait_seconds)


def run_daily_cycle(cv_text: str = "") -> Optional[dict]:
    """Run one idempotent daily cycle."""
    correlation_id = str(uuid4())
    now = datetime.utcnow()
    window_start = now.replace(second=0, microsecond=0)
    if not _claim_run_window("daily", window_start, correlation_id):
        return None

    orchestrator = OrchestratorAgent()
    result_container: dict = {}

    def _invoke() -> None:
        result = orchestrator.execute(cv_text=cv_text, correlation_id=correlation_id)
        result_container.update(result)

    _run_with_retries(_invoke, correlation_id=correlation_id)
    return result_container


def run_weekly_cycle() -> None:
    """Run one idempotent weekly report cycle."""
    correlation_id = str(uuid4())
    now = datetime.utcnow()
    week_start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    if not _claim_run_window("weekly", week_start, correlation_id):
        return

    orchestrator = OrchestratorAgent()

    def _invoke() -> None:
        report = orchestrator.generate_weekly_report()
        logger.info(
            "Weekly report generated: "
            f"opportunities={report.total_opportunities_found}, "
            f"responses={report.total_responses}, interviews={report.total_interviews}"
        )

    _run_with_retries(_invoke, correlation_id=correlation_id)


def start_scheduler() -> None:
    """Start APScheduler with daily and weekly orchestrator jobs."""
    init_db()
    scheduler = BlockingScheduler(timezone=settings.scheduler_timezone)

    scheduler.add_job(
        run_daily_cycle,
        trigger="cron",
        hour=settings.scheduler_daily_hour,
        minute=settings.scheduler_daily_minute,
        id="daily_orchestrator_cycle",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=600,
    )
    scheduler.add_job(
        run_weekly_cycle,
        trigger="cron",
        day_of_week=settings.scheduler_weekly_day,
        hour=settings.scheduler_weekly_hour,
        minute=settings.scheduler_weekly_minute,
        id="weekly_orchestrator_cycle",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=1200,
    )

    logger.info(
        "Scheduler started with daily and weekly jobs: "
        f"daily={settings.scheduler_daily_hour:02d}:{settings.scheduler_daily_minute:02d}, "
        f"weekly={settings.scheduler_weekly_day} "
        f"{settings.scheduler_weekly_hour:02d}:{settings.scheduler_weekly_minute:02d}"
    )
    scheduler.start()

