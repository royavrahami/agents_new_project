"""
Orchestrator Agent — The Central Coordinator.

Responsibilities:
1. Run the daily scan cycle: Intel → Profile → Outreach → Tracker
2. Prioritize actions based on hot_score and pipeline state
3. Deliver a daily briefing and weekly report
4. Send Telegram notifications for hot opportunities
5. Escalate if pipeline is stale (no activity in > 3 days)
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4
from typing import Any, Dict, List, Optional

from config.settings import JobSearchStatus, settings
from core.database import init_db
from core.models import (
    HiddenJob,
    OrchestrationStage,
    StageOutcome,
    WeeklyReport,
)

from .base_agent import BaseAgent
from .coach_agent import CoachAgent
from .intel_agent import IntelAgent
from .outreach_agent import OutreachAgent
from .profile_agent import ProfileAgent
from .tracker_agent import TrackerAgent


class OrchestratorAgent(BaseAgent):
    """
    Master coordinator that manages the full job search pipeline lifecycle.
    Designed to run daily (via cron / Celery beat) or on-demand.

    Args:
        intel:    IntelAgent instance
        profile:  ProfileAgent instance
        outreach: OutreachAgent instance
        tracker:  TrackerAgent instance
        coach:    CoachAgent instance
    """

    def __init__(
        self,
        intel: Optional[IntelAgent] = None,
        profile: Optional[ProfileAgent] = None,
        outreach: Optional[OutreachAgent] = None,
        tracker: Optional[TrackerAgent] = None,
        coach: Optional[CoachAgent] = None,
        max_stage_retries: int = 2,
    ) -> None:
        super().__init__(name="OrchestratorAgent")
        self.intel = intel or IntelAgent()
        self.profile = profile or ProfileAgent()
        self.outreach = outreach or OutreachAgent()
        self.tracker = tracker or TrackerAgent()
        self.coach = coach or CoachAgent()
        self.max_stage_retries = max(1, max_stage_retries)
        self.hot_score_threshold = float(getattr(settings, "hot_job_threshold", 0.6))
        self.escalation_hot_score = float(getattr(settings, "escalation_hot_score", 0.85))
        self.max_outreach_jobs = int(getattr(settings, "outreach_daily_limit", 15))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, cv_text: str = "", **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the full daily scan cycle in the correct order:

        1. Intel Agent — find opportunities & funding events
        2. Tracker Agent — add hot jobs to pipeline
        3. Profile Agent — analyze CV against top opportunities (if CV provided)
        4. Outreach Agent — draft messages for hot jobs
        5. Tracker Agent — pipeline health check
        6. Deliver briefing

        Args:
            cv_text: Optional CV text for profile analysis

        Returns:
            Full briefing dictionary with all agent results
        """
        correlation_id = kwargs.get("correlation_id") or str(uuid4())
        self.logger.info(
            f"=== ORCHESTRATOR: Starting daily cycle (correlation_id={correlation_id}) ==="
        )
        init_db()

        briefing: Dict[str, Any] = {
            "date": datetime.utcnow().isoformat(),
            "candidate": settings.candidate_name,
            "correlation_id": correlation_id,
            "stages": [],
            "escalations": [],
        }

        # ---- Step 1: discover ----
        discover_result = self._run_stage(
            stage=OrchestrationStage.DISCOVER,
            action=lambda: self.intel.execute(correlation_id=correlation_id),
            max_attempts=self.max_stage_retries,
        )
        briefing["stages"].append(discover_result.model_dump(mode="json"))
        intel_result = discover_result.data
        hot_jobs: List[HiddenJob] = intel_result.get("hot_jobs", [])  # type: ignore[assignment]

        briefing["intel"] = {
            "funding_events": intel_result.get("funding_events", 0),
            "hidden_jobs_found": intel_result.get("hidden_jobs", 0),
            "hot_jobs_count": len(hot_jobs),
        }

        # ---- Step 2: prioritize ----
        prioritize_result = self._run_stage(
            stage=OrchestrationStage.PRIORITIZE,
            action=lambda: self._prioritize_jobs(hot_jobs),
            max_attempts=1,
        )
        briefing["stages"].append(prioritize_result.model_dump(mode="json"))
        prioritized_jobs: List[HiddenJob] = prioritize_result.data.get("jobs", [])  # type: ignore[assignment]
        high_priority_jobs = [
            job for job in prioritized_jobs if job.hot_score >= self.escalation_hot_score
        ]
        if high_priority_jobs:
            briefing["escalations"].append(
                (
                    "High-priority opportunities detected. "
                    f"Immediate outreach triggered for {len(high_priority_jobs)} opportunities."
                )
            )

        self.logger.info("Adding prioritized hot jobs to tracker pipeline")
        for job in prioritized_jobs:
            self.tracker.add_opportunity(job)

        # ---- Step 3: tailor ----
        tailor_result = self._run_stage(
            stage=OrchestrationStage.TAILOR,
            action=lambda: self._run_profile_stage(cv_text=cv_text, hot_jobs=prioritized_jobs, correlation_id=correlation_id),
            max_attempts=1,
        )
        briefing["stages"].append(tailor_result.model_dump(mode="json"))
        profile_result = tailor_result.data
        briefing["profile"] = profile_result

        # ---- Step 4: outreach ----
        jobs_for_outreach = high_priority_jobs if high_priority_jobs else prioritized_jobs
        outreach_result_stage = self._run_stage(
            stage=OrchestrationStage.OUTREACH,
            action=lambda: self.outreach.execute(
                hot_jobs=jobs_for_outreach[: self.max_outreach_jobs],
                correlation_id=correlation_id,
            ),
            max_attempts=self.max_stage_retries,
        )
        briefing["stages"].append(outreach_result_stage.model_dump(mode="json"))
        outreach_result = outreach_result_stage.data
        briefing["outreach"] = {
            "messages_drafted": outreach_result.get("messages_drafted", 0),
            "follow_ups_due": len(outreach_result.get("follow_ups_due", [])),
        }

        # ---- Step 5: track ----
        tracker_result_stage = self._run_stage(
            stage=OrchestrationStage.TRACK,
            action=lambda: self.tracker.execute(correlation_id=correlation_id),
            max_attempts=self.max_stage_retries,
        )
        briefing["stages"].append(tracker_result_stage.model_dump(mode="json"))
        tracker_result = tracker_result_stage.data
        briefing["tracker"] = {
            "active_opportunities": tracker_result.get("kpis", {}).get("total_active", 0),
            "bottleneck": tracker_result.get("bottleneck"),
            "ghosted_flagged": tracker_result.get("ghosted_flagged", 0),
            "recommended_actions": tracker_result.get("recommended_actions", []),
        }

        # ---- Step 6: coach ----
        coach_result_stage = self._run_stage(
            stage=OrchestrationStage.COACH,
            action=lambda: self._run_coach_stage(correlation_id=correlation_id),
            max_attempts=1,
        )
        briefing["stages"].append(coach_result_stage.model_dump(mode="json"))
        briefing["coach"] = coach_result_stage.data

        # ---- Step 7: summarize ----
        summarize_stage = self._run_stage(
            stage=OrchestrationStage.SUMMARIZE,
            action=lambda: {"summary_text": self._compose_executive_summary(briefing)},
            max_attempts=1,
        )
        briefing["stages"].append(summarize_stage.model_dump(mode="json"))
        briefing["executive_summary"] = self._compose_executive_summary(briefing)
        briefing["status"] = "ok"

        self.logger.success("=== ORCHESTRATOR: Daily cycle complete ===")
        self._print_briefing(briefing)

        # ---- Notify via Telegram if configured ----
        if settings.telegram_bot_token and settings.telegram_chat_id:
            self._send_telegram_notification(briefing["executive_summary"])

        return briefing

    # ------------------------------------------------------------------
    # Weekly report
    # ------------------------------------------------------------------

    def generate_weekly_report(self) -> WeeklyReport:
        """
        Compose a full weekly report from the Tracker Agent.
        Suitable for stakeholder sharing or personal review.

        Returns:
            WeeklyReport domain object
        """
        report = self.tracker.generate_weekly_report()
        report.top_opportunities = self.intel.get_recent_hot_jobs(limit=5)
        return report

    def print_weekly_report(self) -> None:
        """Print a formatted weekly report to stdout using rich."""
        from rich.console import Console
        from rich.table import Table
        from rich import box

        report = self.generate_weekly_report()
        console = Console()

        console.print(f"\n[bold blue]Weekly Job Search Report[/bold blue]")
        console.print(f"Period: {report.week_start.date()} → {report.week_end.date()}\n")

        # Funnel table
        table = Table(title="Pipeline Funnel", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Opportunities Found", str(report.total_opportunities_found))
        table.add_row("Outreach Sent", str(report.total_outreach_sent))
        table.add_row("Responses Received", str(report.total_responses))
        table.add_row("Interviews", str(report.total_interviews))
        table.add_row("Apply → Response Rate", f"{report.conversion_rate_apply_to_response:.0%}")
        table.add_row("Response → Interview Rate", f"{report.conversion_rate_response_to_interview:.0%}")

        console.print(table)

        if report.bottleneck:
            console.print(f"\n[bold red]Bottleneck:[/bold red] {report.bottleneck}")

        console.print("\n[bold green]Recommended Actions:[/bold green]")
        for i, action in enumerate(report.recommended_actions, 1):
            console.print(f"  {i}. {action}")

        console.print("\n[bold yellow]Top Hot Opportunities:[/bold yellow]")
        for job in report.top_opportunities:
            badge = "🔥" if job.funding_linked else "•"
            console.print(
                f"  {badge} [{job.hot_score:.0%}] {job.role_title} @ {job.company_name}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compose_executive_summary(self, briefing: Dict[str, Any]) -> str:
        """
        Build a concise executive summary from all agent results.
        Suitable for Telegram notification or Slack post.

        Args:
            briefing: Full orchestrator briefing dictionary

        Returns:
            Multi-line summary string
        """
        lines = [
            f"Daily Job Search Briefing - {datetime.utcnow().strftime('%d %b %Y')}",
            f"Candidate: {briefing['candidate']}",
            "",
            f"Intel: {briefing['intel']['hidden_jobs_found']} hidden jobs found "
            f"({briefing['intel']['hot_jobs_count']} HOT), "
            f"{briefing['intel']['funding_events']} funding events",
            f"Outreach: {briefing['outreach']['messages_drafted']} messages drafted, "
            f"{briefing['outreach']['follow_ups_due']} follow-ups due",
            f"Pipeline: {briefing['tracker']['active_opportunities']} active opportunities",
        ]

        if briefing["tracker"]["bottleneck"]:
            lines.append(f"Bottleneck: {briefing['tracker']['bottleneck']}")

        if briefing.get("escalations"):
            lines.append(f"Escalations: {len(briefing['escalations'])}")

        actions = briefing["tracker"]["recommended_actions"]
        if actions:
            lines.append("")
            lines.append("Top Action:")
            lines.append(f" - {actions[0]}")

        return "\n".join(lines)

    def _run_stage(
        self,
        stage: OrchestrationStage,
        action: Any,
        max_attempts: int,
    ) -> StageOutcome:
        """Execute one stage with retries and normalize output to StageOutcome."""
        started_at = datetime.utcnow()
        attempts = 0
        last_error: Optional[str] = None
        normalized_data: Dict[str, Any] = {}

        while attempts < max_attempts:
            attempts += 1
            try:
                result = action()
                if isinstance(result, dict):
                    if result.get("status") in {"error", "failed"}:
                        last_error = str(result.get("error", "Stage returned error status"))
                        normalized_data = result
                    else:
                        normalized_data = result
                        return StageOutcome(
                            stage=stage,
                            status="ok",
                            started_at=started_at,
                            completed_at=datetime.utcnow(),
                            attempts=attempts,
                            data=normalized_data,
                        )
                else:
                    normalized_data = {"result": result}
                    return StageOutcome(
                        stage=stage,
                        status="ok",
                        started_at=started_at,
                        completed_at=datetime.utcnow(),
                        attempts=attempts,
                        data=normalized_data,
                    )
            except Exception as exc:
                last_error = str(exc)
                self.logger.warning(
                    f"Stage {stage.value} failed on attempt {attempts}/{max_attempts}: {exc}"
                )

        status = "skipped" if stage in {OrchestrationStage.TAILOR, OrchestrationStage.COACH} else "error"
        return StageOutcome(
            stage=stage,
            status=status,
            started_at=started_at,
            completed_at=datetime.utcnow(),
            attempts=attempts,
            data=normalized_data,
            error=last_error,
        )

    def _prioritize_jobs(self, hot_jobs: List[HiddenJob]) -> Dict[str, Any]:
        """Sort and filter opportunities by the orchestrator policy."""
        prioritized_jobs = sorted(
            (job for job in hot_jobs if job.hot_score >= self.hot_score_threshold),
            key=lambda job: job.hot_score,
            reverse=True,
        )
        return {
            "status": "ok",
            "jobs": prioritized_jobs,
            "selected_count": len(prioritized_jobs),
            "threshold": self.hot_score_threshold,
        }

    def _run_profile_stage(
        self,
        cv_text: str,
        hot_jobs: List[HiddenJob],
        correlation_id: str,
    ) -> Dict[str, Any]:
        """Run CV tailoring when both CV and target opportunities are available."""
        if not cv_text:
            return {"status": "ok", "note": "No CV provided - profile analysis skipped"}
        if not hot_jobs:
            return {"status": "ok", "note": "No prioritized jobs - profile analysis skipped"}

        top_job = hot_jobs[0]
        result = self.profile.execute(
            cv_text=cv_text,
            job_description=top_job.description_snippet or "",
            correlation_id=correlation_id,
        )
        analysis = result.get("analysis")
        return {
            "status": result.get("status", "ok"),
            "ats_score": analysis.ats_score if hasattr(analysis, "ats_score") else "N/A",
            "top_recommendations": (
                analysis.recommendations[:3]
                if hasattr(analysis, "recommendations")
                else []
            ),
            "target_company": top_job.company_name,
            "target_role": top_job.role_title,
        }

    def _run_coach_stage(self, correlation_id: str) -> Dict[str, Any]:
        """Trigger interview coaching for the top interviewing pipeline entry."""
        interviewing_entries = self.tracker.get_pipeline(
            status=JobSearchStatus.INTERVIEWING,
            limit=1,
        )
        if not interviewing_entries:
            return {"status": "ok", "note": "No interviewing opportunities - coaching skipped"}

        entry = interviewing_entries[0]
        coach_result = self.coach.execute(
            pipeline_entry=entry,
            job_description=entry.notes or "",
            correlation_id=correlation_id,
        )
        prep = coach_result.get("prep")
        questions_count = len(prep.likely_questions) if hasattr(prep, "likely_questions") else 0
        return {
            "status": coach_result.get("status", "ok"),
            "company_name": entry.company_name,
            "role_title": entry.role_title,
            "questions_prepared": questions_count,
            "summary": coach_result.get("summary", ""),
        }

    def _print_briefing(self, briefing: Dict[str, Any]) -> None:
        """Print the briefing to console using rich."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            console = Console()
            console.print(Panel(briefing["executive_summary"], title="Daily Briefing", border_style="blue"))
        except ImportError:
            print(briefing["executive_summary"])

    def _send_telegram_notification(self, message: str) -> None:
        """
        Send a Telegram message to the configured chat.
        Silently skips if credentials are not configured.

        Args:
            message: Text to send
        """
        try:
            import httpx
            url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
            payload = {"chat_id": settings.telegram_chat_id, "text": message, "parse_mode": "HTML"}
            response = httpx.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                self.logger.info("Telegram notification sent successfully")
            else:
                self.logger.warning(f"Telegram notification failed: {response.status_code}")
        except Exception as exc:
            self.logger.warning(f"Telegram notification error: {exc}")
