"""
Main entry point for the Job Search Multi-Agent System.
Run this script to start the daily scan cycle.

Usage:
    python main.py                          # Full daily cycle
    python main.py --report                 # Weekly report only
    python main.py --cv path/to/cv.txt      # With CV analysis
    python main.py --intel-only             # Intel scan only (fastest)
"""

import argparse
import sys
from pathlib import Path

from core.database import init_db
from core.logger import logger
from agents.orchestrator_agent import OrchestratorAgent
from agents.intel_agent import IntelAgent
from orchestration.scheduler_service import start_scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Job Search Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cv", type=str, default="",
        help="Path to CV file (.txt or .md) for profile analysis",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate and print weekly report only",
    )
    parser.add_argument(
        "--intel-only", action="store_true",
        help="Run Intel Agent only (funding scan + hidden jobs)",
    )
    parser.add_argument(
        "--scheduler", action="store_true",
        help="Run APScheduler service for daily/weekly automation",
    )
    parser.add_argument(
        "--show-jobs", action="store_true",
        help="Print all discovered jobs from the database",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    init_db()

    if args.scheduler:
        logger.info("Starting scheduler mode")
        start_scheduler()
        return 0

    if args.show_jobs:
        from rich.console import Console
        from rich.table import Table
        from rich import box
        from core.database import HiddenJobORM, get_db
        with get_db() as db:
            rows = db.query(HiddenJobORM).order_by(
                HiddenJobORM.hot_score.desc(),
                HiddenJobORM.discovered_at.desc()
            ).limit(50).all()
        console = Console()
        table = Table(title=f"Discovered Jobs ({len(rows)} shown)", box=box.ROUNDED)
        table.add_column("#", style="dim", width=3)
        table.add_column("Score", style="yellow", width=6)
        table.add_column("Role", style="cyan", width=32)
        table.add_column("Company", style="green", width=22)
        table.add_column("Source", style="dim", width=16)
        table.add_column("Funded", style="red", width=6)
        table.add_column("URL", style="blue", width=55)
        for i, row in enumerate(rows, 1):
            table.add_row(
                str(i),
                f"{row.hot_score:.2f}" if row.hot_score else "0.00",
                row.role_title[:32] if row.role_title else "",
                row.company_name[:22] if row.company_name else "",
                row.source_domain[:16] if row.source_domain else "",
                "YES" if row.funding_linked else "no",
                row.job_url[:55] if row.job_url else "",
            )
        console.print(table)
        return 0

    if args.intel_only:
        logger.info("Running Intel Agent only")
        agent = IntelAgent()
        result = agent.execute()
        print(result["summary"])
        return 0

    orchestrator = OrchestratorAgent()

    if args.report:
        logger.info("Generating weekly report")
        orchestrator.print_weekly_report()
        return 0

    # Load CV if provided
    cv_text = ""
    if args.cv:
        cv_path = Path(args.cv)
        if not cv_path.exists():
            logger.error(f"CV file not found: {args.cv}")
            return 1
        cv_text = cv_path.read_text(encoding="utf-8")
        logger.info(f"CV loaded from {args.cv} ({len(cv_text)} chars)")

    # Run full daily cycle
    result = orchestrator.execute(cv_text=cv_text)
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nScan interrupted by user (Ctrl+C). Partial results may have been saved.")
        sys.exit(130)
