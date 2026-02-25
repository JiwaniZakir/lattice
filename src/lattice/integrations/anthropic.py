"""
Anthropic provider integration.

Wraps the Anthropic Python SDK for use within the Lattice framework,
providing the same uniform interface as the OpenAI provider.
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field


class CompletionResponse(BaseModel):
    """Standardized completion response."""

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "end_turn"
    latency_ms: float = 0.0
    raw_response: dict[str, Any] = Field(default_factory=dict)


class AnthropicProvider:
    """
    Anthropic API provider with retry and cost tracking.

    Wraps the official anthropic Python package with:
        - Automatic retry
        - Token counting and cost estimation
        - Streaming support
        - Uniform response format matching the OpenAI provider

    Args:
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var).
        default_model: Default model for completions.
        max_retries: Maximum retry attempts.
        timeout_s: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
        timeout_s: float = 120.0,
    ) -> None:
        self._default_model = default_model
        self._max_retries = max_retries
        self._timeout = timeout_s
        self._client = None
        self._api_key = api_key

    def _get_client(self) -> Any:
        """Lazy-initialize the Anthropic async client."""
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(
                api_key=self._api_key,
                max_retries=self._max_retries,
                timeout=self._timeout,
            )
        return self._client

    async def complete(
        self,
        prompt: str | list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system: str | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Generate a completion using Claude.

        Args:
            prompt: String prompt or list of message dicts.
            model: Model to use.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            system: Optional system prompt.
            **kwargs: Additional API parameters.

        Returns:
            CompletionResponse with content and usage.
        """
        client = self._get_client()
        model = model or self._default_model

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        api_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }
        if system:
            api_kwargs["system"] = system

        start = time.perf_counter()
        response = await client.messages.create(**api_kwargs)
        latency = (time.perf_counter() - start) * 1000

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return CompletionResponse(
            content=content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            finish_reason=response.stop_reason or "end_turn",
            latency_ms=latency,
        )

    async def complete_stream(
        self,
        prompt: str | list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming completion.

        Yields:
            Token strings as they arrive.
        """
        client = self._get_client()
        model = model or self._default_model

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        api_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }
        if system:
            api_kwargs["system"] = system

        async with client.messages.stream(**api_kwargs) as stream:
            async for text in stream.text_stream:
                yield text
