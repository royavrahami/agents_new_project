"""
Notifier – sends alerts and report summaries via configured channels:
  - Email (SMTP/TLS)
  - Slack (Bot API)
  - Console (always active – rich formatted output)

Only channels that are fully configured in settings are activated.
All errors are caught and logged – notification failure never crashes the agent.
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.config.settings import settings
from src.storage.models import Trend

logger = logging.getLogger(__name__)
console = Console()


class Notifier:
    """
    Multi-channel notification dispatcher.

    Channels are activated automatically when the required settings are present:
      - Email: requires smtp_user, smtp_password, notify_email
      - Slack: requires slack_bot_token, slack_channel
    """

    def send(
        self,
        alert_trends: list[Trend],
        report_path: Optional[Path] = None,
    ) -> None:
        """
        Dispatch notifications to all configured channels.

        Args:
            alert_trends: Trends that require immediate attention.
            report_path:  Path to the generated HTML report.
        """
        self._console_output(alert_trends, report_path)

        if settings.smtp_user and settings.smtp_password and settings.notify_email:
            self._send_email(alert_trends, report_path)

        if settings.slack_bot_token:
            self._send_slack(alert_trends, report_path)

    # ── Console Output ────────────────────────────────────────────────────────

    @staticmethod
    def _console_output(alert_trends: list[Trend], report_path: Optional[Path]) -> None:
        """Print a rich formatted summary to the terminal."""
        console.print()

        if alert_trends:
            table = Table(
                title="🚨 ALERTS – Immediate Attention Required",
                box=box.ROUNDED,
                style="bold red",
                header_style="bold white on red",
            )
            table.add_column("Trend", style="bold white")
            table.add_column("Category", style="cyan")
            table.add_column("Momentum", style="yellow", justify="right")
            table.add_column("Articles", justify="right")

            for trend in alert_trends:
                table.add_row(
                    trend.name,
                    trend.category,
                    f"{trend.momentum_score:.1f}",
                    str(trend.article_count),
                )
            console.print(table)
        else:
            console.print(Panel(
                "[green]✓ No critical alerts this cycle[/green]",
                title="Alert Status",
                border_style="green",
            ))

        if report_path:
            console.print(Panel(
                f"[bold blue]Report saved to:[/bold blue]\n{report_path}",
                title="📄 Report Generated",
                border_style="blue",
            ))
        console.print()

    # ── Email ─────────────────────────────────────────────────────────────────

    def _send_email(
        self,
        alert_trends: list[Trend],
        report_path: Optional[Path],
    ) -> None:
        """Send an HTML email notification."""
        subject = self._build_email_subject(alert_trends)
        html_body = self._build_email_body(alert_trends, report_path)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = settings.smtp_user
        msg["To"] = settings.notify_email
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        try:
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(settings.smtp_user, settings.smtp_password)
                server.sendmail(settings.smtp_user, settings.notify_email, msg.as_string())
            logger.info("Email notification sent to %s", settings.notify_email)
        except Exception as exc:
            logger.error("Failed to send email notification: %s", exc)

    @staticmethod
    def _build_email_subject(alert_trends: list[Trend]) -> str:
        if alert_trends:
            return f"🚨 QA Intelligence Alert: {len(alert_trends)} trends require attention"
        return "📊 QA Intelligence Report – New update available"

    @staticmethod
    def _build_email_body(alert_trends: list[Trend], report_path: Optional[Path]) -> str:
        alert_section = ""
        if alert_trends:
            items = "".join(
                f"<li><strong>{t.name}</strong> ({t.category}) – "
                f"Momentum: {t.momentum_score:.1f} | Articles: {t.article_count}<br>"
                f"{t.description or ''}</li>"
                for t in alert_trends
            )
            alert_section = f"""
            <h2 style="color:#e94560;">🚨 Alert Trends</h2>
            <ul>{items}</ul>"""

        report_section = ""
        if report_path:
            report_section = f"""
            <p>A full intelligence report has been generated:
            <strong>{report_path.name}</strong></p>"""

        return f"""
        <html><body style="font-family: system-ui, sans-serif; max-width: 600px; margin: 0 auto;">
        <h1 style="color:#0f3460;">🧪 QA Intelligence Update</h1>
        {alert_section}
        {report_section}
        <hr>
        <p style="color:#6b7280; font-size:0.8em;">
        This message was sent by the QA Intelligence Agent running on your system.
        </p>
        </body></html>"""

    # ── Slack ─────────────────────────────────────────────────────────────────

    def _send_slack(
        self,
        alert_trends: list[Trend],
        report_path: Optional[Path],
    ) -> None:
        """Post a Slack message using the Slack SDK."""
        try:
            from slack_sdk import WebClient
            from slack_sdk.errors import SlackApiError
        except ImportError:
            logger.warning("slack_sdk not installed – Slack notifications disabled")
            return

        client = WebClient(token=settings.slack_bot_token)
        blocks = self._build_slack_blocks(alert_trends, report_path)

        try:
            client.chat_postMessage(
                channel=settings.slack_channel,
                blocks=blocks,
                text="QA Intelligence Report",  # Fallback text
            )
            logger.info("Slack notification sent to %s", settings.slack_channel)
        except Exception as exc:
            logger.error("Failed to send Slack notification: %s", exc)

    @staticmethod
    def _build_slack_blocks(alert_trends: list[Trend], report_path: Optional[Path]) -> list[dict]:
        """Build Slack Block Kit message payload."""
        blocks: list[dict] = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🧪 QA Intelligence Update", "emoji": True},
            }
        ]

        if alert_trends:
            alert_text = "\n".join(
                f"• *{t.name}* ({t.category}) — momentum: {t.momentum_score:.1f}"
                for t in alert_trends
            )
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"🚨 *Alert Trends:*\n{alert_text}"},
            })

        if report_path:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"📄 *Report:* `{report_path.name}`"},
            })

        return blocks
