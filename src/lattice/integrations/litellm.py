"""
LiteLLM universal provider integration.

LiteLLM provides a unified interface to 100+ LLM providers. This
integration wraps LiteLLM to provide the same uniform interface
as the OpenAI and Anthropic providers, with automatic fallback
and load balancing across providers.
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


class LiteLLMProvider:
    """
    Universal LLM provider via LiteLLM.

    Supports 100+ LLM providers through a single interface.
    Automatically handles API key routing, retries, and fallbacks.

    Args:
        default_model: Default model identifier.
        api_keys: Optional dict of provider -> API key.
        fallback_models: Models to try if the primary fails.
        max_retries: Maximum retry attempts.
        timeout_s: Request timeout.
    """

    def __init__(
        self,
        default_model: str = "gpt-4o",
        api_keys: dict[str, str] | None = None,
        fallback_models: list[str] | None = None,
        max_retries: int = 3,
        timeout_s: float = 120.0,
    ) -> None:
        self._default_model = default_model
        self._api_keys = api_keys or {}
        self._fallback_models = fallback_models or []
        self._max_retries = max_retries
        self._timeout = timeout_s

        # Set API keys in environment if provided
        if api_keys:
            import os
            for provider, key in api_keys.items():
                env_var = f"{provider.upper()}_API_KEY"
                if env_var not in os.environ:
                    os.environ[env_var] = key

    async def complete(
        self,
        prompt: str | list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Generate a completion via LiteLLM.

        Tries the primary model first, then falls back to configured
        fallback models on failure.

        Args:
            prompt: String prompt or list of message dicts.
            model: Model identifier (e.g., "gpt-4o", "claude-3-5-sonnet").
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            **kwargs: Additional LiteLLM parameters.

        Returns:
            CompletionResponse with content and usage.
        """
        import litellm

        model = model or self._default_model

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        models_to_try = [model] + self._fallback_models
        last_error: Exception | None = None

        for try_model in models_to_try:
            try:
                start = time.perf_counter()
                response = await litellm.acompletion(
                    model=try_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    num_retries=self._max_retries,
                    timeout=self._timeout,
                    **kwargs,
                )
                latency = (time.perf_counter() - start) * 1000

                choice = response.choices[0]
                usage = response.usage

                return CompletionResponse(
                    content=choice.message.content or "",
                    model=response.model or try_model,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                    finish_reason=choice.finish_reason or "stop",
                    latency_ms=latency,
                )
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(
            f"All models failed. Last error: {last_error}"
        )

    async def complete_stream(
        self,
        prompt: str | list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming completion via LiteLLM.

        Yields:
            Token strings as they arrive.
        """
        import litellm

        model = model or self._default_model

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        response = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
            **kwargs,
        )

        async for chunk in response:
            if (
                chunk.choices
                and chunk.choices[0].delta
                and chunk.choices[0].delta.content
            ):
                yield chunk.choices[0].delta.content

    async def embed(
        self,
        text: str,
        model: str = "text-embedding-3-small",
    ) -> Any:
        """Generate an embedding via LiteLLM."""
        import litellm

        response = await litellm.aembedding(model=model, input=[text])
        return type("EmbeddingResponse", (), {
            "embedding": response.data[0]["embedding"],
            "model": response.model,
            "tokens": response.usage.total_tokens if response.usage else 0,
        })()
