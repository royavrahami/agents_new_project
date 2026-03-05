"""Partial failure handling tests for OrchestratorAgent."""

from __future__ import annotations

from types import SimpleNamespace

from agents.orchestrator_agent import OrchestratorAgent
from core.models import HiddenJob


class TestOrchestratorPartialFailure:
    """Verifies one failing stage does not crash the full orchestrator cycle."""

    def test_outreach_failure_is_captured_in_stage_outcome(self) -> None:
        hot_job = HiddenJob(
            company_name="Acme",
            role_title="QA Manager",
            job_url="https://example.com/jobs/1",
            source_domain="comeet.com",
            hot_score=0.9,
        )
        intel = SimpleNamespace(
            execute=lambda **kwargs: {
                "status": "ok",
                "funding_events": 0,
                "hidden_jobs": 1,
                "hot_jobs": [hot_job],
            }
        )
        profile = SimpleNamespace(execute=lambda **kwargs: {"status": "ok", "analysis": None})

        def failing_outreach(**kwargs):
            raise RuntimeError("network timeout")

        outreach = SimpleNamespace(execute=failing_outreach)
        tracker = SimpleNamespace(
            add_opportunity=lambda job: None,
            execute=lambda **kwargs: {
                "status": "ok",
                "kpis": {"total_active": 1},
                "bottleneck": "Low response rate",
                "ghosted_flagged": 0,
                "recommended_actions": ["Refine message"],
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

        outreach_stage = next(stage for stage in result["stages"] if stage["stage"] == "outreach")
        assert outreach_stage["status"] == "error"
        assert outreach_stage["attempts"] == 2
        assert "network timeout" in outreach_stage["error"]
        assert result["status"] == "ok"

