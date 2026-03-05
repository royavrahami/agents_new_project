"""Happy-path tests for OrchestratorAgent."""

from __future__ import annotations

from types import SimpleNamespace

from agents.orchestrator_agent import OrchestratorAgent
from core.models import HiddenJob


class TestOrchestratorHappyPath:
    """Verifies the default daily cycle succeeds end-to-end."""

    def test_execute_runs_all_core_stages(self) -> None:
        hot_job = HiddenJob(
            company_name="Acme",
            role_title="QA Manager",
            job_url="https://example.com/jobs/1",
            source_domain="comeet.com",
            hot_score=0.9,
            description_snippet="Lead quality strategy",
        )
        intel = SimpleNamespace(
            execute=lambda **kwargs: {
                "status": "ok",
                "funding_events": 1,
                "hidden_jobs": 1,
                "hot_jobs": [hot_job],
            }
        )
        profile = SimpleNamespace(
            execute=lambda **kwargs: {
                "status": "ok",
                "analysis": SimpleNamespace(ats_score=88.0, recommendations=["Improve summary", "Add KPIs"]),
            }
        )
        outreach = SimpleNamespace(
            execute=lambda **kwargs: {
                "status": "ok",
                "messages_drafted": 1,
                "follow_ups_due": [],
            }
        )
        tracker = SimpleNamespace(
            add_opportunity=lambda job: None,
            execute=lambda **kwargs: {
                "status": "ok",
                "kpis": {"total_active": 3},
                "bottleneck": None,
                "ghosted_flagged": 0,
                "recommended_actions": ["Follow up with 2 applications"],
            },
            get_pipeline=lambda **kwargs: [],
            generate_weekly_report=lambda: None,
        )
        coach = SimpleNamespace(execute=lambda **kwargs: {"status": "ok", "summary": "ready"})

        orchestrator = OrchestratorAgent(
            intel=intel,
            profile=profile,
            outreach=outreach,
            tracker=tracker,
            coach=coach,
        )
        result = orchestrator.execute(cv_text="sample cv")

        assert result["status"] == "ok"
        assert result["intel"]["hot_jobs_count"] == 1
        assert result["profile"]["ats_score"] == 88.0
        assert result["outreach"]["messages_drafted"] == 1
        assert result["tracker"]["active_opportunities"] == 3
        assert "executive_summary" in result
        stage_names = [stage["stage"] for stage in result["stages"]]
        assert stage_names == [
            "discover",
            "prioritize",
            "tailor",
            "outreach",
            "track",
            "coach",
            "summarize",
        ]

