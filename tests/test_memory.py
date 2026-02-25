"""Tests for scoped shared memory."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from lattice.core.memory import ScopedMemory


@pytest.fixture
def memory() -> ScopedMemory:
    return ScopedMemory(default_scope="test")


class TestMemoryBasicOps:
    @pytest.mark.asyncio
    async def test_set_and_get(self, memory: ScopedMemory) -> None:
        await memory.set("key1", "value1")
        result = await memory.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_missing_key(self, memory: ScopedMemory) -> None:
        result = await memory.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_overwrites(self, memory: ScopedMemory) -> None:
        await memory.set("key1", "first")
        await memory.set("key1", "second")
        result = await memory.get("key1")
        assert result == "second"

    @pytest.mark.asyncio
    async def test_delete(self, memory: ScopedMemory) -> None:
        await memory.set("key1", "value1")
        deleted = await memory.delete("key1")
        assert deleted is True
        result = await memory.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, memory: ScopedMemory) -> None:
        deleted = await memory.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_complex_values(self, memory: ScopedMemory) -> None:
        data = {"nested": {"key": [1, 2, 3]}, "flag": True}
        await memory.set("complex", data)
        result = await memory.get("complex")
        assert result == data


class TestMemoryScoping:
    @pytest.mark.asyncio
    async def test_create_scope(self, memory: ScopedMemory) -> None:
        scope = memory.create_scope("child", parent="test")
        assert scope.name == "child"
        assert scope.parent == "test"

    @pytest.mark.asyncio
    async def test_scope_isolation(self, memory: ScopedMemory) -> None:
        memory.create_scope("scope_a")
        memory.create_scope("scope_b")

        await memory.set("key", "a_value", scope="scope_a")
        await memory.set("key", "b_value", scope="scope_b")

        result_a = await memory.get("key", scope="scope_a", inherit=False)
        result_b = await memory.get("key", scope="scope_b", inherit=False)

        assert result_a == "a_value"
        assert result_b == "b_value"

    @pytest.mark.asyncio
    async def test_scope_inheritance(self, memory: ScopedMemory) -> None:
        memory.create_scope("child", parent="test")
        await memory.set("parent_key", "from_parent", scope="test")

        result = await memory.get("parent_key", scope="child", inherit=True)
        assert result == "from_parent"

    @pytest.mark.asyncio
    async def test_no_inheritance(self, memory: ScopedMemory) -> None:
        memory.create_scope("child", parent="test")
        await memory.set("parent_key", "from_parent", scope="test")

        result = await memory.get("parent_key", scope="child", inherit=False)
        assert result is None

    def test_invalid_parent(self, memory: ScopedMemory) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            memory.create_scope("orphan", parent="nonexistent")

    def test_duplicate_scope(self, memory: ScopedMemory) -> None:
        scope1 = memory.create_scope("dup")
        scope2 = memory.create_scope("dup")  # Should return existing
        assert scope1.name == scope2.name


class TestMemoryTTL:
    @pytest.mark.asyncio
    async def test_ttl_expiration(self, memory: ScopedMemory) -> None:
        await memory.set("expiring", "value", ttl_s=0.01)
        await asyncio.sleep(0.02)
        result = await memory.get("expiring")
        assert result is None

    @pytest.mark.asyncio
    async def test_ttl_not_expired(self, memory: ScopedMemory) -> None:
        await memory.set("fresh", "value", ttl_s=10.0)
        result = await memory.get("fresh")
        assert result == "value"


class TestMemorySimilaritySearch:
    @pytest.mark.asyncio
    async def test_search_similar(self, memory: ScopedMemory) -> None:
        # Store entries with embeddings
        emb_a = [1.0, 0.0, 0.0]
        emb_b = [0.0, 1.0, 0.0]
        emb_c = [0.9, 0.1, 0.0]  # Similar to emb_a

        await memory.set("a", "value_a", embedding=emb_a)
        await memory.set("b", "value_b", embedding=emb_b)
        await memory.set("c", "value_c", embedding=emb_c)

        results = await memory.search_similar(emb_a, top_k=2, threshold=0.5)
        assert len(results) >= 1
        # "a" should be most similar to itself, then "c"
        keys = [r.key for r in results]
        assert "a" in keys

    @pytest.mark.asyncio
    async def test_search_no_embeddings(self, memory: ScopedMemory) -> None:
        await memory.set("no_emb", "value")
        results = await memory.search_similar([1.0, 0.0, 0.0])
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_zero_vector(self, memory: ScopedMemory) -> None:
        results = await memory.search_similar([0.0, 0.0, 0.0])
        assert len(results) == 0


class TestMemoryCAS:
    @pytest.mark.asyncio
    async def test_compare_and_swap_success(self, memory: ScopedMemory) -> None:
        await memory.set("counter", 1)
        swapped = await memory.compare_and_swap("counter", 1, 2)
        assert swapped is True
        assert await memory.get("counter") == 2

    @pytest.mark.asyncio
    async def test_compare_and_swap_failure(self, memory: ScopedMemory) -> None:
        await memory.set("counter", 1)
        swapped = await memory.compare_and_swap("counter", 99, 2)
        assert swapped is False
        assert await memory.get("counter") == 1

    @pytest.mark.asyncio
    async def test_cas_missing_key(self, memory: ScopedMemory) -> None:
        swapped = await memory.compare_and_swap("missing", 1, 2)
        assert swapped is False


class TestMemoryStats:
    @pytest.mark.asyncio
    async def test_stats(self, memory: ScopedMemory) -> None:
        await memory.set("k1", "v1")
        await memory.set("k2", "v2")
        stats = memory.get_stats()
        assert stats["total_entries"] == 2
        assert stats["num_scopes"] >= 1

    @pytest.mark.asyncio
    async def test_list_keys(self, memory: ScopedMemory) -> None:
        await memory.set("alpha", 1)
        await memory.set("beta", 2)
        await memory.set("gamma", 3)

        keys = await memory.list_keys()
        assert set(keys) == {"alpha", "beta", "gamma"}

    @pytest.mark.asyncio
    async def test_list_keys_pattern(self, memory: ScopedMemory) -> None:
        await memory.set("task_1", 1)
        await memory.set("task_2", 2)
        await memory.set("result_1", 3)

        keys = await memory.list_keys(pattern="task_*")
        assert set(keys) == {"task_1", "task_2"}

    @pytest.mark.asyncio
    async def test_clear(self, memory: ScopedMemory) -> None:
        await memory.set("k1", "v1")
        await memory.set("k2", "v2")
        count = await memory.clear()
        assert count == 2
        assert await memory.get("k1") is None
