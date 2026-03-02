"""
Tracker Agent — Job Search CRM & KPI Dashboard.

Responsibilities:
1. Maintain the job pipeline (IDENTIFIED → APPLIED → … → OFFER)
2. Detect bottlenecks (where is the funnel dropping off?)
3. Flag GHOSTED entries (no activity for N days)
4. Compute weekly KPIs: conversion rates, velocity, pipeline health
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from config.settings import JobSearchStatus, settings
from core.database import PipelineEntryORM, get_db
from core.logger import logger
from core.models import HiddenJob, PipelineEntry, WeeklyReport

from .base_agent import BaseAgent


# Days without activity before an entry is flagged as GHOSTED
_GHOST_THRESHOLD_DAYS = 7

# Pipeline stage order (used for funnel analysis)
_STAGE_ORDER = [
    JobSearchStatus.IDENTIFIED,
    JobSearchStatus.APPLIED,
    JobSearchStatus.RESPONDED,
    JobSearchStatus.INTERVIEWING,
    JobSearchStatus.OFFER,
]


class TrackerAgent(BaseAgent):
    """
    CRM layer for the job search pipeline.
    Tracks every opportunity from discovery to offer/rejection,
    and surfaces actionable KPIs.
    """

    def __init__(self) -> None:
        super().__init__(name="TrackerAgent")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Perform a full pipeline health check:
        1. Auto-flag GHOSTED entries
        2. Compute KPIs
        3. Identify bottlenecks
        4. Return a structured weekly report

        Returns:
            {
                "status": "ok",
                "kpis": Dict,
                "bottleneck": str | None,
                "ghosted_flagged": int,
                "recommended_actions": List[str],
                "summary": str
            }
        """
        self.logger.info("Running pipeline health check")

        # Step 1 — Auto-flag ghosts
        ghosted_count = self._flag_ghosted_entries()

        # Step 2 — Compute KPIs
        kpis = self.compute_kpis()

        # Step 3 — Find bottleneck
        bottleneck = self._identify_bottleneck(kpis)

        # Step 4 — Generate action recommendations
        actions = self._generate_recommendations(kpis, bottleneck)

        summary = (
            f"Pipeline health: {kpis['total_active']} active opportunities. "
            f"Response rate: {kpis['response_rate']:.0%}. "
            f"Bottleneck: {bottleneck or 'None'}."
        )

        self.logger.success(summary)
        return {
            "status": "ok",
            "kpis": kpis,
            "bottleneck": bottleneck,
            "ghosted_flagged": ghosted_count,
            "recommended_actions": actions,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Pipeline CRUD
    # ------------------------------------------------------------------

    def add_opportunity(self, job: HiddenJob) -> PipelineEntry:
        """
        Add a new job opportunity to the pipeline at IDENTIFIED stage.

        Args:
            job: HiddenJob from Intel Agent

        Returns:
            The created PipelineEntry
        """
        entry = PipelineEntry(
            company_name=job.company_name,
            role_title=job.role_title,
            job_url=job.job_url,
            status=JobSearchStatus.IDENTIFIED,
            hot_score=job.hot_score,
        )
        with get_db() as db:
            # Avoid duplicates
            existing = (
                db.query(PipelineEntryORM)
                .filter_by(job_url=job.job_url)
                .first()
            )
            if existing:
                self.logger.debug(f"Pipeline entry already exists for {job.job_url}")
                return entry

            orm = PipelineEntryORM(
                id=str(entry.id),
                company_name=entry.company_name,
                role_title=entry.role_title,
                job_url=entry.job_url,
                status=entry.status.value,
                hot_score=entry.hot_score,
                last_activity_at=datetime.utcnow(),
            )
            db.add(orm)
            self.logger.info(f"Added to pipeline: {entry.company_name} — {entry.role_title}")
        return entry

    def advance_stage(
        self,
        entry_id: str,
        new_status: JobSearchStatus,
        notes: Optional[str] = None,
    ) -> None:
        """
        Move an opportunity to the next pipeline stage.

        Args:
            entry_id:   UUID string of the PipelineEntry
            new_status: Target JobSearchStatus
            notes:      Optional notes (e.g. interview feedback)
        """
        with get_db() as db:
            entry = db.query(PipelineEntryORM).filter_by(id=entry_id).first()
            if not entry:
                logger.warning(f"Pipeline entry not found: {entry_id}")
                return
            entry.status = new_status.value
            entry.last_activity_at = datetime.utcnow()
            if notes:
                entry.notes = notes
            if new_status == JobSearchStatus.APPLIED:
                entry.applied_at = datetime.utcnow()
            self.logger.info(
                f"Stage advanced: {entry.company_name} → {new_status.value}"
            )

    def reject(self, entry_id: str, reason: Optional[str] = None) -> None:
        """Mark an entry as REJECTED with an optional reason."""
        with get_db() as db:
            entry = db.query(PipelineEntryORM).filter_by(id=entry_id).first()
            if entry:
                entry.status = JobSearchStatus.REJECTED.value
                entry.rejection_reason = reason
                entry.last_activity_at = datetime.utcnow()

    def get_pipeline(
        self,
        status: Optional[JobSearchStatus] = None,
        limit: int = 100,
    ) -> List[PipelineEntry]:
        """
        Retrieve pipeline entries, optionally filtered by status.

        Args:
            status: Filter by stage (None = all)
            limit:  Maximum number of entries

        Returns:
            List of PipelineEntry domain objects
        """
        results = []
        with get_db() as db:
            query = db.query(PipelineEntryORM)
            if status:
                query = query.filter_by(status=status.value)
            rows = query.order_by(PipelineEntryORM.last_activity_at.desc()).limit(limit).all()
            # Convert to Pydantic models *inside* the session to avoid DetachedInstanceError
            for r in rows:
                results.append(PipelineEntry(
                    company_name=r.company_name,
                    role_title=r.role_title,
                    job_url=r.job_url,
                    status=JobSearchStatus(r.status),
                    applied_at=r.applied_at,
                    last_activity_at=r.last_activity_at or datetime.utcnow(),
                    notes=r.notes,
                    rejection_reason=r.rejection_reason,
                    interview_stage=r.interview_stage,
                    offer_amount=r.offer_amount,
                    hot_score=r.hot_score or 0.0,
                ))
        return results

    # ------------------------------------------------------------------
    # KPIs & Analytics
    # ------------------------------------------------------------------

    def compute_kpis(self) -> Dict[str, Any]:
        """
        Compute all key performance indicators for the pipeline.

        KPIs computed:
        - Pipeline counts per stage
        - Response rate (APPLIED → RESPONDED)
        - Interview rate (RESPONDED → INTERVIEWING)
        - Offer rate (INTERVIEWING → OFFER)
        - Ghost rate (APPLIED → GHOSTED)
        - Average days to response

        Returns:
            Dictionary of KPI names to values
        """
        counts: Dict[str, int] = {s.value: 0 for s in JobSearchStatus}
        with get_db() as db:
            all_entries = db.query(PipelineEntryORM).all()
            # Collect status strings inside session to avoid DetachedInstanceError
            statuses = [e.status for e in all_entries]
            total = len(all_entries)

        for s in statuses:
            counts[s] = counts.get(s, 0) + 1

        total_applied = counts["APPLIED"] + counts["RESPONDED"] + counts["INTERVIEWING"] + counts["OFFER"] + counts["REJECTED"] + counts["GHOSTED"]
        total_responded = counts["RESPONDED"] + counts["INTERVIEWING"] + counts["OFFER"]
        total_interview = counts["INTERVIEWING"] + counts["OFFER"]
        total_active = sum(
            counts[s] for s in ["IDENTIFIED", "APPLIED", "RESPONDED", "INTERVIEWING"]
        )

        return {
            "total_in_pipeline": total,
            "total_active": total_active,
            "by_stage": counts,
            "response_rate": total_responded / max(total_applied, 1),
            "interview_rate": total_interview / max(total_responded, 1),
            "offer_rate": counts["OFFER"] / max(total_interview, 1),
            "ghost_rate": counts["GHOSTED"] / max(total_applied, 1),
            "rejection_rate": counts["REJECTED"] / max(total_applied, 1),
            "computed_at": datetime.utcnow().isoformat(),
        }

    def _identify_bottleneck(self, kpis: Dict[str, Any]) -> Optional[str]:
        """
        Identify the weakest stage in the funnel.

        Logic: The stage with the lowest conversion rate below 0.3 is the bottleneck.

        Args:
            kpis: Output of compute_kpis()

        Returns:
            Human-readable bottleneck description or None
        """
        thresholds = {
            "response_rate": (0.20, "Low response rate — improve CV/outreach message quality"),
            "interview_rate": (0.30, "Low interview rate — work on phone screen / LinkedIn profile"),
            "offer_rate": (0.25, "Low offer rate — focus on interview preparation"),
            "ghost_rate": (0.40, "High ghost rate — follow up more aggressively"),
        }
        for metric, (threshold, message) in thresholds.items():
            value = kpis.get(metric, 0)
            if metric == "ghost_rate":
                if value > threshold:
                    return message
            else:
                if value < threshold and kpis.get("total_active", 0) > 3:
                    return message
        return None

    def _generate_recommendations(
        self, kpis: Dict[str, Any], bottleneck: Optional[str]
    ) -> List[str]:
        """
        Generate 3-5 concrete actions based on pipeline state.

        Args:
            kpis:       Computed KPIs
            bottleneck: Identified bottleneck stage

        Returns:
            List of actionable recommendation strings
        """
        actions = []
        by_stage = kpis.get("by_stage", {})

        if by_stage.get("IDENTIFIED", 0) > 10:
            actions.append(
                f"You have {by_stage['IDENTIFIED']} unactioned opportunities — apply to the top 5 this week"
            )
        if by_stage.get("APPLIED", 0) < 5:
            actions.append("Low application volume — target 5+ new applications this week")
        if kpis.get("response_rate", 1.0) < 0.15:
            actions.append("Response rate < 15% — A/B test a new outreach message format")
        if by_stage.get("GHOSTED", 0) > 3:
            actions.append(f"Send follow-ups to {by_stage['GHOSTED']} ghosted applications")
        if kpis.get("offer_rate", 0) == 0 and by_stage.get("INTERVIEWING", 0) > 0:
            actions.append("In interviews but no offers — book a mock interview session")
        if bottleneck:
            actions.insert(0, f"TOP PRIORITY: {bottleneck}")

        return actions[:5]

    # ------------------------------------------------------------------
    # Ghost detection
    # ------------------------------------------------------------------

    def _flag_ghosted_entries(self) -> int:
        """
        Automatically move APPLIED entries with no activity for > 7 days to GHOSTED.

        Returns:
            Number of entries flagged as GHOSTED
        """
        cutoff = datetime.utcnow() - timedelta(days=_GHOST_THRESHOLD_DAYS)
        count = 0
        with get_db() as db:
            stale = (
                db.query(PipelineEntryORM)
                .filter(
                    PipelineEntryORM.status == JobSearchStatus.APPLIED.value,
                    PipelineEntryORM.last_activity_at < cutoff,
                )
                .all()
            )
            for entry in stale:
                entry.status = JobSearchStatus.GHOSTED.value
                count += 1
                logger.debug(f"Flagged as GHOSTED: {entry.company_name}")

        if count:
            self.logger.info(f"Flagged {count} entries as GHOSTED (no activity > {_GHOST_THRESHOLD_DAYS}d)")
        return count

    # ------------------------------------------------------------------
    # Weekly report
    # ------------------------------------------------------------------

    def generate_weekly_report(self) -> WeeklyReport:
        """
        Generate a full weekly summary for the Orchestrator / Stakeholder view.

        Returns:
            WeeklyReport domain object
        """
        now = datetime.utcnow()
        week_start = now - timedelta(days=7)
        kpis = self.compute_kpis()
        bottleneck = self._identify_bottleneck(kpis)
        actions = self._generate_recommendations(kpis, bottleneck)
        by_stage = kpis["by_stage"]

        return WeeklyReport(
            week_start=week_start,
            week_end=now,
            total_opportunities_found=kpis["total_in_pipeline"],
            total_outreach_sent=by_stage.get("APPLIED", 0),
            total_responses=by_stage.get("RESPONDED", 0),
            total_interviews=by_stage.get("INTERVIEWING", 0),
            conversion_rate_apply_to_response=kpis["response_rate"],
            conversion_rate_response_to_interview=kpis["interview_rate"],
            bottleneck=bottleneck,
            recommended_actions=actions,
        )
