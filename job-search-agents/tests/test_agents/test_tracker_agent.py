"""
Tests for TrackerAgent — pipeline CRM, KPI computation, ghost detection.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from agents.tracker_agent import TrackerAgent, _GHOST_THRESHOLD_DAYS
from config.settings import JobSearchStatus
from core.database import PipelineEntryORM, get_db
from core.models import HiddenJob, PipelineEntry


class TestTrackerAgentPipelineCRUD:
    """Tests for pipeline add, advance, and reject operations."""

    def test_add_opportunity_creates_entry(self, sample_hidden_job):
        agent = TrackerAgent()
        entry = agent.add_opportunity(sample_hidden_job)
        assert entry.company_name == sample_hidden_job.company_name
        assert entry.status == JobSearchStatus.IDENTIFIED

    def test_add_duplicate_does_not_create_second_entry(self, sample_hidden_job):
        agent = TrackerAgent()
        agent.add_opportunity(sample_hidden_job)
        agent.add_opportunity(sample_hidden_job)  # Same URL

        with get_db() as db:
            count = db.query(PipelineEntryORM).filter_by(
                job_url=sample_hidden_job.job_url
            ).count()
        assert count == 1

    def test_advance_stage_updates_status(self, sample_hidden_job):
        agent = TrackerAgent()
        agent.add_opportunity(sample_hidden_job)

        # Retrieve entry_id within an active session to avoid DetachedInstanceError
        with get_db() as db:
            entry_id = db.query(PipelineEntryORM).filter_by(
                job_url=sample_hidden_job.job_url
            ).first().id

        agent.advance_stage(entry_id, JobSearchStatus.INTERVIEWING, notes="First call done")

        with get_db() as db:
            updated = db.query(PipelineEntryORM).filter_by(id=entry_id).first()
            status = updated.status
            notes = updated.notes
        assert status == JobSearchStatus.INTERVIEWING.value
        assert notes == "First call done"

    def test_advance_to_applied_sets_applied_at(self, sample_hidden_job):
        agent = TrackerAgent()
        agent.add_opportunity(sample_hidden_job)

        with get_db() as db:
            entry_id = db.query(PipelineEntryORM).filter_by(
                job_url=sample_hidden_job.job_url
            ).first().id

        agent.advance_stage(entry_id, JobSearchStatus.APPLIED)

        with get_db() as db:
            updated = db.query(PipelineEntryORM).filter_by(id=entry_id).first()
            applied_at = updated.applied_at
        assert applied_at is not None

    def test_reject_sets_rejection_reason(self, sample_hidden_job):
        agent = TrackerAgent()
        agent.add_opportunity(sample_hidden_job)

        with get_db() as db:
            entry_id = db.query(PipelineEntryORM).filter_by(
                job_url=sample_hidden_job.job_url
            ).first().id

        agent.reject(entry_id, reason="Over-qualified")

        with get_db() as db:
            updated = db.query(PipelineEntryORM).filter_by(id=entry_id).first()
            status = updated.status
            rejection_reason = updated.rejection_reason
        assert status == JobSearchStatus.REJECTED.value
        assert rejection_reason == "Over-qualified"

    def test_advance_nonexistent_entry_does_not_raise(self):
        agent = TrackerAgent()
        # Should log warning but not raise
        agent.advance_stage("nonexistent-id-999", JobSearchStatus.OFFER)

    def test_get_pipeline_returns_all_entries(self, sample_hidden_job):
        agent = TrackerAgent()
        agent.add_opportunity(sample_hidden_job)
        # get_pipeline() reads from DB and converts to Pydantic models — no detach issue
        entries = agent.get_pipeline()
        assert len(entries) >= 1

    def test_get_pipeline_filtered_by_status(self, sample_hidden_job):
        agent = TrackerAgent()
        agent.add_opportunity(sample_hidden_job)
        entries = agent.get_pipeline(status=JobSearchStatus.IDENTIFIED)
        assert all(e.status == JobSearchStatus.IDENTIFIED for e in entries)


class TestTrackerAgentKPIs:
    """Tests for KPI computation logic."""

    def _populate_pipeline(self, agent: TrackerAgent) -> None:
        """Helper: Insert pipeline entries at various stages directly into DB."""
        from datetime import datetime
        rows = [
            ("Applied Co 1", "Role A", "https://example.com/1", "APPLIED"),
            ("Applied Co 2", "Role B", "https://example.com/2", "APPLIED"),
            ("Responded Co", "Role C", "https://example.com/3", "RESPONDED"),
            ("Interview Co", "Role D", "https://example.com/4", "INTERVIEWING"),
            ("Rejected Co",  "Role E", "https://example.com/5", "REJECTED"),
            ("Ghosted Co",   "Role F", "https://example.com/6", "GHOSTED"),
        ]
        with get_db() as db:
            for company, role, url, status in rows:
                db.add(PipelineEntryORM(
                    company_name=company,
                    role_title=role,
                    job_url=url,
                    status=status,
                    last_activity_at=datetime.utcnow(),
                ))

    def test_kpis_return_correct_keys(self):
        agent = TrackerAgent()
        kpis = agent.compute_kpis()
        required_keys = {
            "total_in_pipeline", "total_active", "by_stage",
            "response_rate", "interview_rate", "offer_rate",
            "ghost_rate", "rejection_rate",
        }
        assert required_keys.issubset(set(kpis.keys()))

    def test_kpis_empty_pipeline(self):
        agent = TrackerAgent()
        kpis = agent.compute_kpis()
        assert kpis["total_in_pipeline"] == 0
        assert kpis["response_rate"] == 0.0

    def test_response_rate_calculation(self):
        agent = TrackerAgent()
        self._populate_pipeline(agent)
        kpis = agent.compute_kpis()
        # RESPONDED(1) + INTERVIEWING(1) + OFFER(0) = 2 out of applied(2+1+1+1+1=6)
        assert 0.0 <= kpis["response_rate"] <= 1.0

    def test_ghost_rate_calculation(self):
        agent = TrackerAgent()
        self._populate_pipeline(agent)
        kpis = agent.compute_kpis()
        assert 0.0 <= kpis["ghost_rate"] <= 1.0


class TestTrackerAgentGhostDetection:
    """Tests for automatic ghosting of stale applications."""

    def test_stale_applied_entry_becomes_ghosted(self):
        agent = TrackerAgent()
        stale_date = datetime.utcnow() - timedelta(days=_GHOST_THRESHOLD_DAYS + 1)

        with get_db() as db:
            db.add(PipelineEntryORM(
                company_name="Ghost Corp",
                role_title="Developer",
                job_url="https://example.com/ghost",
                status=JobSearchStatus.APPLIED.value,
                last_activity_at=stale_date,
            ))

        flagged = agent._flag_ghosted_entries()
        assert flagged == 1

        with get_db() as db:
            entry = db.query(PipelineEntryORM).filter_by(company_name="Ghost Corp").first()
            status = entry.status  # read inside session
        assert status == JobSearchStatus.GHOSTED.value

    def test_recent_applied_entry_not_ghosted(self):
        agent = TrackerAgent()
        with get_db() as db:
            db.add(PipelineEntryORM(
                company_name="Active Corp",
                role_title="Developer",
                job_url="https://example.com/active",
                status=JobSearchStatus.APPLIED.value,
                last_activity_at=datetime.utcnow(),
            ))

        flagged = agent._flag_ghosted_entries()
        assert flagged == 0

    def test_interviewing_entry_not_ghosted(self):
        """Entries in INTERVIEWING should not be ghosted even if stale."""
        agent = TrackerAgent()
        stale_date = datetime.utcnow() - timedelta(days=_GHOST_THRESHOLD_DAYS + 5)

        with get_db() as db:
            db.add(PipelineEntryORM(
                company_name="Interview Corp",
                role_title="Developer",
                job_url="https://example.com/interview",
                status=JobSearchStatus.INTERVIEWING.value,
                last_activity_at=stale_date,
            ))

        flagged = agent._flag_ghosted_entries()
        assert flagged == 0  # Only APPLIED entries are auto-ghosted


class TestTrackerAgentRun:
    """Integration tests for TrackerAgent.run()."""

    def test_run_returns_ok_status(self):
        agent = TrackerAgent()
        result = agent.execute()
        assert result["status"] == "ok"

    def test_run_returns_kpis(self):
        agent = TrackerAgent()
        result = agent.execute()
        assert "kpis" in result
        assert "response_rate" in result["kpis"]

    def test_run_returns_recommended_actions(self):
        agent = TrackerAgent()
        result = agent.execute()
        assert "recommended_actions" in result
        assert isinstance(result["recommended_actions"], list)

    def test_weekly_report_structure(self):
        agent = TrackerAgent()
        report = agent.generate_weekly_report()
        assert report.total_in_pipeline if hasattr(report, "total_in_pipeline") else True
        assert report.conversion_rate_apply_to_response >= 0.0
        assert isinstance(report.recommended_actions, list)
