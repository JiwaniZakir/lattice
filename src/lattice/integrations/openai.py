"""
OpenAI provider integration.

Wraps the OpenAI Python SDK for use within the Lattice framework,
providing a uniform interface for completions, embeddings, and
streaming with automatic retry, token counting, and cost tracking.
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
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    raw_response: dict[str, Any] = Field(default_factory=dict)


class EmbeddingResponse(BaseModel):
    """Standardized embedding response."""

    embedding: list[float]
    model: str
    tokens: int = 0


class OpenAIProvider:
    """
    OpenAI API provider with retry and cost tracking.

    Wraps the official openai Python package with:
        - Automatic retry with exponential backoff
        - Token counting and cost estimation
        - Streaming support
        - Uniform response format

    Args:
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
        default_model: Default model for completions.
        max_retries: Maximum retry attempts.
        timeout_s: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "gpt-4o",
        max_retries: int = 3,
        timeout_s: float = 120.0,
    ) -> None:
        self._default_model = default_model
        self._max_retries = max_retries
        self._timeout = timeout_s
        self._client = None
        self._api_key = api_key

    def _get_client(self) -> Any:
        """Lazy-initialize the OpenAI async client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
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
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Generate a completion.

        Args:
            prompt: String prompt or list of message dicts.
            model: Model to use (defaults to configured model).
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
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

        start = time.perf_counter()
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        latency = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        usage = response.usage

        return CompletionResponse(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            finish_reason=choice.finish_reason or "stop",
            latency_ms=latency,
        )

    async def complete_stream(
        self,
        prompt: str | list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
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

        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
            **kwargs,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def embed(
        self,
        text: str,
        model: str = "text-embedding-3-small",
    ) -> EmbeddingResponse:
        """
        Generate an embedding.

        Args:
            text: Text to embed.
            model: Embedding model to use.

        Returns:
            EmbeddingResponse with embedding vector.
        """
        client = self._get_client()
        response = await client.embeddings.create(
            model=model,
            input=text,
        )

        return EmbeddingResponse(
            embedding=response.data[0].embedding,
            model=response.model,
            tokens=response.usage.total_tokens if response.usage else 0,
        )
