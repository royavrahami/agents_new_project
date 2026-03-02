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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    init_db()

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
    sys.exit(main())
