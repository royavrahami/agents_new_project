"""Escalation behavior tests for OrchestratorAgent."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agents.orchestrator_agent import OrchestratorAgent
from core.models import HiddenJob


class TestOrchestratorEscalation:
    """Verifies high-score opportunities trigger immediate outreach prioritization."""

    def test_high_priority_jobs_are_outreached_first(self) -> None:
        high = HiddenJob(
            company_name="HotStartup",
            role_title="QA Director",
            job_url="https://example.com/jobs/high",
            source_domain="lever.co",
            hot_score=0.95,
        )
        regular = HiddenJob(
            company_name="RegularCo",
            role_title="QA Lead",
            job_url="https://example.com/jobs/regular",
            source_domain="comeet.com",
            hot_score=0.65,
        )
        intel = SimpleNamespace(
            execute=lambda **kwargs: {
                "status": "ok",
                "funding_events": 1,
                "hidden_jobs": 2,
                "hot_jobs": [regular, high],
            }
        )
        profile = SimpleNamespace(execute=lambda **kwargs: {"status": "ok", "analysis": None})
        outreach = MagicMock()
        outreach.execute.return_value = {"status": "ok", "messages_drafted": 1, "follow_ups_due": []}
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
        )
        result = orchestrator.execute(cv_text="")

        outreach_jobs = outreach.execute.call_args.kwargs["hot_jobs"]
        assert len(outreach_jobs) == 1
        assert outreach_jobs[0].company_name == "HotStartup"
        assert len(result["escalations"]) == 1

