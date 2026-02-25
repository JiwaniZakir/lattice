"""
Structured logging for the Lattice framework.

Provides JSON-structured logging with context propagation across
the async execution pipeline. Integrates with OpenTelemetry trace
context so log entries include trace and span IDs.

Uses structlog for structured, machine-parseable log output that
is still human-readable in development mode.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from pydantic import BaseModel, Field


class LogConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "json"  # "json" or "console"
    include_trace_context: bool = True
    log_file: str | None = None


class StructuredLogger:
    """
    Structured logger with context propagation.

    Wraps Python's standard logging with structured output formatting.
    In JSON mode, outputs machine-parseable log lines. In console mode,
    outputs human-readable colored output.

    Args:
        name: Logger name.
        config: Logging configuration.
    """

    def __init__(
        self,
        name: str = "lattice",
        config: LogConfig | None = None,
    ) -> None:
        self._config = config or LogConfig()
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, self._config.level.upper()))
        self._context: dict[str, Any] = {}

        self._setup_handler()

    def _setup_handler(self) -> None:
        """Configure the logging handler and formatter."""
        # Remove existing handlers
        self._logger.handlers.clear()

        if self._config.format == "json":
            formatter = _JSONFormatter()
        else:
            formatter = _ConsoleFormatter()

        if self._config.log_file:
            handler = logging.FileHandler(self._config.log_file)
        else:
            handler = logging.StreamHandler(sys.stdout)

        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

    def bind(self, **kwargs: Any) -> StructuredLogger:
        """
        Create a child logger with additional context fields.

        Args:
            **kwargs: Context fields to bind.

        Returns:
            A new StructuredLogger with the bound context.
        """
        child = StructuredLogger.__new__(StructuredLogger)
        child._config = self._config
        child._logger = self._logger
        child._context = {**self._context, **kwargs}
        return child

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log at DEBUG level."""
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log at INFO level."""
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log at WARNING level."""
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        """Log at ERROR level."""
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        """Log at CRITICAL level."""
        self._log(logging.CRITICAL, msg, **kwargs)

    def _log(self, level: int, msg: str, **kwargs: Any) -> None:
        """Internal log method that merges context."""
        extra = {**self._context, **kwargs}

        # Add trace context if available and configured
        if self._config.include_trace_context:
            trace_ctx = self._get_trace_context()
            if trace_ctx:
                extra.update(trace_ctx)

        self._logger.log(level, msg, extra={"structured": extra})

    @staticmethod
    def _get_trace_context() -> dict[str, str]:
        """Extract OpenTelemetry trace context if available."""
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            ctx = span.get_span_context()
            if ctx and ctx.trace_id:
                return {
                    "trace_id": format(ctx.trace_id, "032x"),
                    "span_id": format(ctx.span_id, "016x"),
                }
        except (ImportError, Exception):
            pass
        return {}


class _JSONFormatter(logging.Formatter):
    """JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        import time

        output = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            ),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }

        structured = getattr(record, "structured", None)
        if structured:
            output.update(structured)

        return json.dumps(output, default=str)


class _ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[31;1m",  # Bold red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET

        parts = [
            f"{color}[{record.levelname:>8}]{reset}",
            record.getMessage(),
        ]

        structured = getattr(record, "structured", None)
        if structured:
            ctx_parts = [f"{k}={v}" for k, v in structured.items()]
            if ctx_parts:
                parts.append(f"  ({', '.join(ctx_parts)})")

        return " ".join(parts)
