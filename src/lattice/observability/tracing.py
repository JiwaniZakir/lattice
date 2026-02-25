"""
OpenTelemetry integration for distributed tracing.

Provides automatic span creation for all Lattice operations, including:

    - Per-plan execution spans
    - Per-step agent invocation spans
    - Router decision spans
    - Verifier check spans
    - LLM call spans with token/cost attributes

Spans include rich attributes (agent_id, tokens, cost, strategy) for
debugging and performance analysis in tools like Jaeger or Honeycomb.
"""

from __future__ import annotations

import functools
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Iterator

from pydantic import BaseModel, Field


class SpanAttributes(BaseModel):
    """Standard span attributes for Lattice operations."""

    lattice_operation: str = ""
    lattice_agent_id: str = ""
    lattice_plan_id: str = ""
    lattice_step_id: str = ""
    lattice_model: str = ""
    lattice_tokens: int = 0
    lattice_cost_usd: float = 0.0
    lattice_strategy: str = ""
    lattice_success: bool = True
    lattice_error: str = ""


class TracingConfig(BaseModel):
    """Configuration for tracing."""

    service_name: str = "lattice"
    endpoint: str = "http://localhost:4317"
    enabled: bool = True
    sample_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    export_format: str = "otlp"  # "otlp", "jaeger", "console"


class TracingManager:
    """
    OpenTelemetry tracing manager for Lattice.

    Initializes the OpenTelemetry tracer provider and provides
    context managers for creating spans with Lattice-specific
    attributes.

    Args:
        config: Tracing configuration.
    """

    def __init__(self, config: TracingConfig | None = None) -> None:
        self._config = config or TracingConfig()
        self._tracer = None
        self._provider = None

        if self._config.enabled:
            self._setup_tracer()

    def _setup_tracer(self) -> None:
        """Initialize the OpenTelemetry tracer."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import (
                BatchSpanProcessor,
                ConsoleSpanExporter,
            )

            resource = Resource.create({
                "service.name": self._config.service_name,
                "service.version": "0.1.0",
            })

            self._provider = TracerProvider(resource=resource)

            if self._config.export_format == "console":
                processor = BatchSpanProcessor(ConsoleSpanExporter())
            else:
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                        OTLPSpanExporter,
                    )
                    exporter = OTLPSpanExporter(endpoint=self._config.endpoint)
                    processor = BatchSpanProcessor(exporter)
                except ImportError:
                    processor = BatchSpanProcessor(ConsoleSpanExporter())

            self._provider.add_span_processor(processor)
            trace.set_tracer_provider(self._provider)
            self._tracer = trace.get_tracer(
                "lattice",
                schema_url="https://opentelemetry.io/schemas/1.21.0",
            )
        except ImportError:
            self._tracer = None

    @asynccontextmanager
    async def span(
        self,
        name: str,
        attributes: SpanAttributes | None = None,
        **extra_attrs: Any,
    ) -> AsyncIterator[Any]:
        """
        Create an async span for an operation.

        Args:
            name: Span name.
            attributes: Lattice span attributes.
            **extra_attrs: Additional attributes.

        Yields:
            The OpenTelemetry span (or a no-op if tracing is disabled).
        """
        if self._tracer is None:
            yield _NoOpSpan()
            return

        attrs = self._build_attributes(attributes, extra_attrs)

        with self._tracer.start_as_current_span(name, attributes=attrs) as otel_span:
            try:
                yield otel_span
            except Exception as e:
                otel_span.set_attribute("lattice.error", str(e))
                otel_span.set_attribute("lattice.success", False)
                otel_span.record_exception(e)
                raise

    @contextmanager
    def sync_span(
        self,
        name: str,
        attributes: SpanAttributes | None = None,
        **extra_attrs: Any,
    ) -> Iterator[Any]:
        """Create a synchronous span."""
        if self._tracer is None:
            yield _NoOpSpan()
            return

        attrs = self._build_attributes(attributes, extra_attrs)

        with self._tracer.start_as_current_span(name, attributes=attrs) as otel_span:
            try:
                yield otel_span
            except Exception as e:
                otel_span.set_attribute("lattice.error", str(e))
                otel_span.record_exception(e)
                raise

    def _build_attributes(
        self,
        attrs: SpanAttributes | None,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert SpanAttributes and extras into a flat dict."""
        result: dict[str, Any] = {}
        if attrs:
            for key, value in attrs.model_dump().items():
                if value:
                    result[f"lattice.{key.removeprefix('lattice_')}"] = value
        for key, value in extra.items():
            result[f"lattice.{key}"] = value
        return result

    def shutdown(self) -> None:
        """Flush and shut down the tracer provider."""
        if self._provider:
            self._provider.shutdown()


class _NoOpSpan:
    """No-op span for when tracing is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass
