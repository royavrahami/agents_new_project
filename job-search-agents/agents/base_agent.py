"""
Abstract base class for all Job Search Agents.
Enforces a consistent interface: every agent exposes a `run()` method
and reports its state via structured logging.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

from core.logger import logger


class BaseAgent(ABC):
    """
    Base class that every agent extends.

    Responsibilities of subclasses:
    - Override `run()` with domain-specific logic
    - Raise exceptions for unrecoverable errors (orchestrator handles them)
    - Use self.logger for all output (never print())

    Args:
        name: Human-readable agent name used in logs
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = logger.bind(agent=name)
        self._run_count: int = 0
        self._last_run: datetime | None = None

    @abstractmethod
    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the agent's main task.

        Returns:
            A dictionary summarising the run result.
            Convention: {"status": "ok"|"error", "data": ..., "summary": "..."}
        """

    def _before_run(self) -> None:
        """Hook called automatically before `run()`. Override to add pre-run logic."""
        self._run_count += 1
        self._last_run = datetime.utcnow()
        self.logger.info(f"[{self.name}] Starting run #{self._run_count}")

    def _after_run(self, result: Dict[str, Any]) -> None:
        """Hook called automatically after `run()`. Override to add post-run logic."""
        status = result.get("status", "unknown")
        self.logger.info(f"[{self.name}] Completed run #{self._run_count} — status={status}")

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Public entry point. Wraps `run()` with lifecycle hooks and top-level error handling.
        Orchestrator calls `execute()`, not `run()` directly.
        """
        correlation_id = kwargs.pop("correlation_id", None)
        if correlation_id:
            with self.logger.contextualize(correlation_id=correlation_id):
                self._before_run()
                try:
                    result = self.run(**kwargs)
                except Exception as exc:
                    self.logger.exception(f"[{self.name}] Unhandled error during run: {exc}")
                    result = {"status": "error", "error": str(exc), "data": None}
                self._after_run(result)
                return result

        self._before_run()
        try:
            result = self.run(**kwargs)
        except Exception as exc:
            self.logger.exception(f"[{self.name}] Unhandled error during run: {exc}")
            result = {"status": "error", "error": str(exc), "data": None}
        self._after_run(result)
        return result

    @property
    def stats(self) -> Dict[str, Any]:
        """Return basic runtime statistics for this agent."""
        return {
            "name": self.name,
            "run_count": self._run_count,
            "last_run": self._last_run.isoformat() if self._last_run else None,
        }
