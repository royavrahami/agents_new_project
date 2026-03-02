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

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.settings import settings
from core.database import init_db
from core.logger import logger
from core.models import HiddenJob, WeeklyReport

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
    ) -> None:
        super().__init__(name="OrchestratorAgent")
        self.intel = intel or IntelAgent()
        self.profile = profile or ProfileAgent()
        self.outreach = outreach or OutreachAgent()
        self.tracker = tracker or TrackerAgent()
        self.coach = coach or CoachAgent()

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
        self.logger.info("=== ORCHESTRATOR: Starting daily cycle ===")
        init_db()

        briefing: Dict[str, Any] = {
            "date": datetime.utcnow().isoformat(),
            "candidate": settings.candidate_name,
        }

        # ---- Step 1: Intel scan ----
        self.logger.info("Step 1/5 — Intel scan")
        intel_result = self.intel.execute()
        briefing["intel"] = {
            "funding_events": intel_result.get("funding_events", 0),
            "hidden_jobs_found": intel_result.get("hidden_jobs", 0),
            "hot_jobs_count": len(intel_result.get("hot_jobs", [])),
        }
        hot_jobs: List[HiddenJob] = intel_result.get("hot_jobs", [])

        # ---- Step 2: Add hot jobs to tracker pipeline ----
        self.logger.info("Step 2/5 — Adding hot jobs to pipeline")
        for job in hot_jobs:
            self.tracker.add_opportunity(job)

        # ---- Step 3: Profile analysis (if CV provided) ----
        self.logger.info("Step 3/5 — Profile analysis")
        if cv_text and hot_jobs:
            top_job = hot_jobs[0]
            profile_result = self.profile.execute(
                cv_text=cv_text,
                job_description=top_job.description_snippet or "",
            )
            briefing["profile"] = {
                "ats_score": profile_result.get("analysis", {}).ats_score
                if hasattr(profile_result.get("analysis"), "ats_score")
                else "N/A",
                "top_recommendations": (
                    profile_result.get("analysis").recommendations[:3]
                    if hasattr(profile_result.get("analysis"), "recommendations")
                    else []
                ),
            }
        else:
            briefing["profile"] = {"note": "No CV provided — skipped profile analysis"}

        # ---- Step 4: Outreach drafting ----
        self.logger.info("Step 4/5 — Outreach drafting")
        outreach_result = self.outreach.execute(hot_jobs=hot_jobs)
        briefing["outreach"] = {
            "messages_drafted": outreach_result.get("messages_drafted", 0),
            "follow_ups_due": len(outreach_result.get("follow_ups_due", [])),
        }

        # ---- Step 5: Tracker health check ----
        self.logger.info("Step 5/5 — Tracker health check")
        tracker_result = self.tracker.execute()
        briefing["tracker"] = {
            "active_opportunities": tracker_result.get("kpis", {}).get("total_active", 0),
            "bottleneck": tracker_result.get("bottleneck"),
            "ghosted_flagged": tracker_result.get("ghosted_flagged", 0),
            "recommended_actions": tracker_result.get("recommended_actions", []),
        }

        # ---- Compose executive briefing ----
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
            f"📅 Daily Job Search Briefing — {datetime.utcnow().strftime('%d %b %Y')}",
            f"👤 Candidate: {briefing['candidate']}",
            "",
            f"🔍 Intel: {briefing['intel']['hidden_jobs_found']} hidden jobs found "
            f"({briefing['intel']['hot_jobs_count']} HOT), "
            f"{briefing['intel']['funding_events']} funding events",
            f"📤 Outreach: {briefing['outreach']['messages_drafted']} messages drafted, "
            f"{briefing['outreach']['follow_ups_due']} follow-ups due",
            f"📊 Pipeline: {briefing['tracker']['active_opportunities']} active opportunities",
        ]

        if briefing["tracker"]["bottleneck"]:
            lines.append(f"⚠️ Bottleneck: {briefing['tracker']['bottleneck']}")

        actions = briefing["tracker"]["recommended_actions"]
        if actions:
            lines.append("")
            lines.append("✅ Top Action:")
            lines.append(f"   → {actions[0]}")

        return "\n".join(lines)

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
