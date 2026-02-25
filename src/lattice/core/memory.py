"""
Scoped shared memory with hierarchical namespaces and TTL support.

Provides a multi-layer memory system for agents to share state during
execution. Memory is organized into scopes (namespaces) that support:

    - Hierarchical access: child scopes inherit from parent scopes
    - TTL-based expiration
    - Redis-backed persistence for distributed deployments
    - In-memory cache for single-process usage
    - Atomic read-modify-write operations
    - Memory size tracking and eviction

Inspired by the Voyager skill library (Wang et al., 2023), the memory
system also supports semantic search over stored entries when an
embedding function is provided.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """A single entry in scoped memory."""

    key: str
    value: Any
    scope: str
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    ttl_s: float | None = None
    access_count: int = 0
    embedding: list[float] | None = None

    @property
    def is_expired(self) -> bool:
        """Check if this entry has exceeded its TTL."""
        if self.ttl_s is None:
            return False
        return (time.time() - self.updated_at) > self.ttl_s


class MemoryScope(BaseModel):
    """A namespace in the memory hierarchy."""

    name: str
    parent: str | None = None
    entries: dict[str, MemoryEntry] = Field(default_factory=dict)
    children: list[str] = Field(default_factory=list)
    max_entries: int = 10_000


class ScopedMemory:
    """
    Hierarchical scoped memory system for multi-agent state sharing.

    Memory is organized into scopes (namespaces). Each scope can have
    a parent scope, forming a tree. Key lookups traverse up the scope
    hierarchy if not found in the current scope.

    Features:
        - Hierarchical scoping with inheritance
        - TTL-based automatic expiration
        - Optional Redis backend for distributed deployments
        - Semantic similarity search via embeddings
        - Atomic compare-and-swap updates
        - Memory usage tracking

    Args:
        default_scope: The root scope name.
        redis_url: Optional Redis URL for persistent storage.
        max_entries_per_scope: Maximum entries per scope.
    """

    def __init__(
        self,
        default_scope: str = "global",
        redis_url: str | None = None,
        max_entries_per_scope: int = 10_000,
    ) -> None:
        self._default_scope = default_scope
        self._max_entries = max_entries_per_scope
        self._scopes: dict[str, MemoryScope] = {
            default_scope: MemoryScope(
                name=default_scope,
                max_entries=max_entries_per_scope,
            )
        }
        self._lock = asyncio.Lock()
        self._redis = None
        self._redis_url = redis_url

    async def _get_redis(self) -> Any:
        """Lazy-initialize Redis connection."""
        if self._redis is None and self._redis_url:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(
                    self._redis_url, decode_responses=True
                )
            except ImportError:
                pass
        return self._redis

    def create_scope(
        self,
        name: str,
        parent: str | None = None,
    ) -> MemoryScope:
        """
        Create a new memory scope.

        Args:
            name: Scope name (must be unique).
            parent: Optional parent scope name for hierarchical lookup.

        Returns:
            The created MemoryScope.

        Raises:
            ValueError: If scope already exists or parent doesn't exist.
        """
        if name in self._scopes:
            return self._scopes[name]

        if parent and parent not in self._scopes:
            raise ValueError(f"Parent scope '{parent}' does not exist")

        scope = MemoryScope(
            name=name,
            parent=parent,
            max_entries=self._max_entries,
        )
        self._scopes[name] = scope

        if parent:
            self._scopes[parent].children.append(name)

        return scope

    async def set(
        self,
        key: str,
        value: Any,
        scope: str | None = None,
        ttl_s: float | None = None,
        embedding: list[float] | None = None,
    ) -> MemoryEntry:
        """
        Store a value in memory.

        Args:
            key: The key to store under.
            value: The value to store (must be JSON-serializable).
            scope: The scope to store in (defaults to the root scope).
            ttl_s: Optional time-to-live in seconds.
            embedding: Optional embedding vector for semantic search.

        Returns:
            The created or updated MemoryEntry.
        """
        scope_name = scope or self._default_scope

        async with self._lock:
            if scope_name not in self._scopes:
                self.create_scope(scope_name)

            mem_scope = self._scopes[scope_name]

            # Evict expired entries if at capacity
            if len(mem_scope.entries) >= mem_scope.max_entries:
                self._evict_expired(mem_scope)

            # Evict LRU if still at capacity
            if len(mem_scope.entries) >= mem_scope.max_entries:
                self._evict_lru(mem_scope)

            now = time.time()
            if key in mem_scope.entries:
                entry = mem_scope.entries[key]
                entry.value = value
                entry.updated_at = now
                entry.ttl_s = ttl_s
                if embedding is not None:
                    entry.embedding = embedding
            else:
                entry = MemoryEntry(
                    key=key,
                    value=value,
                    scope=scope_name,
                    created_at=now,
                    updated_at=now,
                    ttl_s=ttl_s,
                    embedding=embedding,
                )
                mem_scope.entries[key] = entry

        # Persist to Redis if available
        redis = await self._get_redis()
        if redis:
            redis_key = f"lattice:mem:{scope_name}:{key}"
            await redis.set(
                redis_key,
                json.dumps(entry.model_dump(), default=str),
                ex=int(ttl_s) if ttl_s else None,
            )

        return entry

    async def get(
        self,
        key: str,
        scope: str | None = None,
        inherit: bool = True,
    ) -> Any | None:
        """
        Retrieve a value from memory.

        If the key is not found in the specified scope and inherit is True,
        traverses up the scope hierarchy to find it.

        Args:
            key: The key to look up.
            scope: The scope to search in.
            inherit: Whether to search parent scopes.

        Returns:
            The stored value, or None if not found.
        """
        scope_name = scope or self._default_scope

        current_scope = scope_name
        while current_scope is not None:
            if current_scope not in self._scopes:
                break

            mem_scope = self._scopes[current_scope]
            if key in mem_scope.entries:
                entry = mem_scope.entries[key]
                if entry.is_expired:
                    del mem_scope.entries[key]
                else:
                    entry.access_count += 1
                    return entry.value

            if not inherit:
                break
            current_scope = mem_scope.parent

        # Try Redis fallback
        redis = await self._get_redis()
        if redis:
            redis_key = f"lattice:mem:{scope_name}:{key}"
            data = await redis.get(redis_key)
            if data:
                entry_data = json.loads(data)
                return entry_data.get("value")

        return None

    async def delete(self, key: str, scope: str | None = None) -> bool:
        """Delete a key from memory. Returns True if the key existed."""
        scope_name = scope or self._default_scope

        async with self._lock:
            if scope_name in self._scopes:
                if key in self._scopes[scope_name].entries:
                    del self._scopes[scope_name].entries[key]

                    redis = await self._get_redis()
                    if redis:
                        await redis.delete(f"lattice:mem:{scope_name}:{key}")
                    return True
        return False

    async def search_similar(
        self,
        query_embedding: list[float] | np.ndarray,
        scope: str | None = None,
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> list[MemoryEntry]:
        """
        Search memory entries by embedding similarity.

        Uses cosine similarity to find entries with embeddings close to
        the query embedding. Only entries with stored embeddings are
        considered.

        Args:
            query_embedding: The query embedding vector.
            scope: Scope to search in (searches all if None).
            top_k: Maximum number of results.
            threshold: Minimum similarity threshold.

        Returns:
            List of matching MemoryEntry objects, sorted by similarity.
        """
        if isinstance(query_embedding, list):
            query_vec = np.array(query_embedding, dtype=np.float64)
        else:
            query_vec = query_embedding.astype(np.float64)

        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_vec = query_vec / query_norm

        candidates: list[tuple[float, MemoryEntry]] = []

        scopes_to_search = (
            [scope] if scope else list(self._scopes.keys())
        )

        for scope_name in scopes_to_search:
            if scope_name not in self._scopes:
                continue
            for entry in self._scopes[scope_name].entries.values():
                if entry.is_expired or entry.embedding is None:
                    continue

                entry_vec = np.array(entry.embedding, dtype=np.float64)
                entry_norm = np.linalg.norm(entry_vec)
                if entry_norm == 0:
                    continue
                entry_vec = entry_vec / entry_norm

                similarity = float(np.dot(query_vec, entry_vec))
                if similarity >= threshold:
                    candidates.append((similarity, entry))

        # Sort by similarity descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in candidates[:top_k]]

    async def list_keys(
        self, scope: str | None = None, pattern: str | None = None
    ) -> list[str]:
        """List all keys in a scope, optionally filtered by pattern."""
        scope_name = scope or self._default_scope
        if scope_name not in self._scopes:
            return []

        keys = list(self._scopes[scope_name].entries.keys())
        if pattern:
            import fnmatch
            keys = [k for k in keys if fnmatch.fnmatch(k, pattern)]
        return keys

    async def compare_and_swap(
        self,
        key: str,
        expected_value: Any,
        new_value: Any,
        scope: str | None = None,
    ) -> bool:
        """
        Atomic compare-and-swap operation.

        Only updates the value if the current value matches expected_value.

        Returns:
            True if the swap was performed.
        """
        scope_name = scope or self._default_scope

        async with self._lock:
            if scope_name not in self._scopes:
                return False

            entries = self._scopes[scope_name].entries
            if key not in entries:
                return False

            entry = entries[key]
            if entry.is_expired:
                del entries[key]
                return False

            if entry.value == expected_value:
                entry.value = new_value
                entry.updated_at = time.time()
                return True
            return False

    def get_stats(self, scope: str | None = None) -> dict[str, Any]:
        """Get memory usage statistics."""
        if scope:
            scopes = [scope] if scope in self._scopes else []
        else:
            scopes = list(self._scopes.keys())

        total_entries = 0
        total_expired = 0
        scope_stats: dict[str, dict[str, int]] = {}

        for s in scopes:
            mem_scope = self._scopes[s]
            entries = len(mem_scope.entries)
            expired = sum(1 for e in mem_scope.entries.values() if e.is_expired)
            total_entries += entries
            total_expired += expired
            scope_stats[s] = {
                "entries": entries,
                "expired": expired,
                "max_entries": mem_scope.max_entries,
                "children": len(mem_scope.children),
            }

        return {
            "total_entries": total_entries,
            "total_expired": total_expired,
            "num_scopes": len(scopes),
            "scopes": scope_stats,
        }

    def _evict_expired(self, scope: MemoryScope) -> int:
        """Remove expired entries from a scope. Returns count removed."""
        expired_keys = [
            k for k, v in scope.entries.items() if v.is_expired
        ]
        for k in expired_keys:
            del scope.entries[k]
        return len(expired_keys)

    def _evict_lru(self, scope: MemoryScope, count: int = 1) -> int:
        """Remove least-recently-used entries. Returns count removed."""
        if not scope.entries:
            return 0

        # Sort by access count then by updated_at
        sorted_entries = sorted(
            scope.entries.items(),
            key=lambda x: (x[1].access_count, x[1].updated_at),
        )

        removed = 0
        for key, _ in sorted_entries[:count]:
            del scope.entries[key]
            removed += 1

        return removed

    async def clear(self, scope: str | None = None) -> int:
        """Clear all entries in a scope (or all scopes). Returns count cleared."""
        async with self._lock:
            if scope:
                if scope in self._scopes:
                    count = len(self._scopes[scope].entries)
                    self._scopes[scope].entries.clear()
                    return count
                return 0
            else:
                count = sum(
                    len(s.entries) for s in self._scopes.values()
                )
                for s in self._scopes.values():
                    s.entries.clear()
                return count
