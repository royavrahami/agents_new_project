"""Retry behavior tests for OrchestratorAgent."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agents.orchestrator_agent import OrchestratorAgent
from core.models import HiddenJob


class TestOrchestratorRetry:
    """Verifies stage retry logic for recoverable failures."""

    def test_discover_stage_retries_and_recovers(self) -> None:
        hot_job = HiddenJob(
            company_name="Acme",
            role_title="QA Manager",
            job_url="https://example.com/jobs/1",
            source_domain="comeet.com",
            hot_score=0.9,
        )
        intel = MagicMock()
        intel.execute.side_effect = [
            {"status": "error", "error": "temporary"},
            {"status": "ok", "funding_events": 0, "hidden_jobs": 1, "hot_jobs": [hot_job]},
        ]
        profile = SimpleNamespace(execute=lambda **kwargs: {"status": "ok", "analysis": None})
        outreach = SimpleNamespace(execute=lambda **kwargs: {"status": "ok", "messages_drafted": 1, "follow_ups_due": []})
        tracker = SimpleNamespace(
            add_opportunity=lambda job: None,
            execute=lambda **kwargs: {
                "status": "ok",
                "kpis": {"total_active": 1},
                "bottleneck": None,
                "ghosted_flagged": 0,
                "recommended_actions": [],
            },
            get_pipeline=lambda **kwargs: [],
            generate_weekly_report=lambda: None,
        )
        coach = SimpleNamespace(execute=lambda **kwargs: {"status": "ok"})

        orchestrator = OrchestratorAgent(
            intel=intel,
            profile=profile,
            outreach=outreach,
            tracker=tracker,
            coach=coach,
            max_stage_retries=2,
        )
        result = orchestrator.execute(cv_text="")

        discover_stage = next(stage for stage in result["stages"] if stage["stage"] == "discover")
        assert discover_stage["status"] == "ok"
        assert discover_stage["attempts"] == 2
        assert intel.execute.call_count == 2

