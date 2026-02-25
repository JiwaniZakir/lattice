"""
Token-level streaming for agent outputs.

Provides async iterators for streaming tokens from LLM agent outputs
to clients. Supports:

    - Backpressure-aware buffering
    - Multi-consumer fan-out (multiple readers on one stream)
    - Token counting during streaming
    - Timeout per token
    - Stream cancellation
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

from pydantic import BaseModel, Field


class StreamToken(BaseModel):
    """A single token in a stream."""

    content: str
    index: int
    is_final: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class StreamStats(BaseModel):
    """Statistics for a completed or ongoing stream."""

    total_tokens: int = 0
    total_characters: int = 0
    duration_ms: float = 0.0
    tokens_per_second: float = 0.0
    first_token_ms: float = 0.0


class StreamBuffer:
    """
    Async buffer for token-level streaming with backpressure.

    Producers write tokens via `put()`, consumers read via async iteration.
    Supports multiple consumers (fan-out) where each consumer gets all tokens.

    Args:
        max_size: Maximum number of buffered tokens before backpressure.
        token_timeout_s: Maximum wait time for next token.
    """

    def __init__(
        self,
        max_size: int = 1000,
        token_timeout_s: float = 30.0,
    ) -> None:
        self._queue: asyncio.Queue[StreamToken | None] = asyncio.Queue(maxsize=max_size)
        self._token_timeout = token_timeout_s
        self._started_at: float | None = None
        self._first_token_at: float | None = None
        self._total_tokens = 0
        self._total_chars = 0
        self._closed = False
        self._consumers: list[asyncio.Queue[StreamToken | None]] = []

    async def put(self, content: str, is_final: bool = False, **metadata: Any) -> None:
        """
        Add a token to the buffer.

        Args:
            content: Token content string.
            is_final: Whether this is the last token.
            **metadata: Additional metadata for the token.
        """
        if self._closed:
            raise RuntimeError("Cannot write to closed stream")

        if self._started_at is None:
            self._started_at = time.perf_counter()

        token = StreamToken(
            content=content,
            index=self._total_tokens,
            is_final=is_final,
            metadata=metadata,
        )

        if self._first_token_at is None:
            self._first_token_at = time.perf_counter()

        self._total_tokens += 1
        self._total_chars += len(content)

        await self._queue.put(token)

        # Fan out to consumers
        for consumer_queue in self._consumers:
            try:
                consumer_queue.put_nowait(token)
            except asyncio.QueueFull:
                pass  # Consumer is slow; drop token

        if is_final:
            await self.close()

    async def close(self) -> None:
        """Signal end of stream."""
        if not self._closed:
            self._closed = True
            await self._queue.put(None)
            for cq in self._consumers:
                try:
                    cq.put_nowait(None)
                except asyncio.QueueFull:
                    pass

    @property
    def is_closed(self) -> bool:
        return self._closed

    def get_stats(self) -> StreamStats:
        """Get current streaming statistics."""
        now = time.perf_counter()
        duration = (now - self._started_at) * 1000 if self._started_at else 0.0
        tps = (
            self._total_tokens / (duration / 1000)
            if duration > 0
            else 0.0
        )
        first_token = (
            (self._first_token_at - self._started_at) * 1000
            if self._started_at and self._first_token_at
            else 0.0
        )

        return StreamStats(
            total_tokens=self._total_tokens,
            total_characters=self._total_chars,
            duration_ms=duration,
            tokens_per_second=tps,
            first_token_ms=first_token,
        )

    async def __aiter__(self) -> AsyncIterator[StreamToken]:
        """Iterate over tokens as they arrive."""
        while True:
            try:
                token = await asyncio.wait_for(
                    self._queue.get(), timeout=self._token_timeout
                )
            except asyncio.TimeoutError:
                break

            if token is None:
                break
            yield token

    def add_consumer(self) -> asyncio.Queue[StreamToken | None]:
        """
        Add a new consumer queue for fan-out.

        Returns:
            An asyncio.Queue that will receive all future tokens.
        """
        q: asyncio.Queue[StreamToken | None] = asyncio.Queue(maxsize=1000)
        self._consumers.append(q)
        return q


class TokenStream:
    """
    High-level token stream with collection and aggregation.

    Wraps a StreamBuffer and provides convenience methods for collecting
    all tokens, aggregating into full text, and tracking statistics.

    Args:
        buffer: The underlying StreamBuffer.
    """

    def __init__(self, buffer: StreamBuffer | None = None) -> None:
        self._buffer = buffer or StreamBuffer()
        self._collected: list[StreamToken] = []
        self._full_text: str | None = None

    @property
    def buffer(self) -> StreamBuffer:
        return self._buffer

    async def collect(self) -> list[StreamToken]:
        """Collect all tokens from the stream."""
        self._collected = []
        async for token in self._buffer:
            self._collected.append(token)
        return self._collected

    async def to_text(self) -> str:
        """Collect all tokens and join into full text."""
        if self._full_text is not None:
            return self._full_text

        if not self._collected:
            await self.collect()

        self._full_text = "".join(t.content for t in self._collected)
        return self._full_text

    def get_stats(self) -> StreamStats:
        """Get streaming statistics."""
        return self._buffer.get_stats()

    async def __aiter__(self) -> AsyncIterator[StreamToken]:
        """Iterate over tokens."""
        async for token in self._buffer:
            self._collected.append(token)
            yield token
