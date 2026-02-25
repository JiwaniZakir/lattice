"""
Execution checkpointing for fault recovery.

Saves execution state at configurable intervals so that failed
executions can be resumed from the last successful step rather
than restarted from scratch. Supports:

    - File-based checkpoints (JSON on disk)
    - Redis-based checkpoints (distributed)
    - Configurable checkpoint frequency
    - Checkpoint cleanup and retention policies
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from lattice.core.executor import ExecutionResult


class Checkpoint(BaseModel):
    """A saved execution checkpoint."""

    plan_id: str
    execution_id: str
    state: dict[str, Any]
    step_index: int
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointConfig(BaseModel):
    """Configuration for the checkpoint manager."""

    backend: str = "file"  # "file" or "redis"
    checkpoint_dir: str = "./checkpoints"
    redis_url: str | None = None
    max_checkpoints_per_plan: int = 10
    retention_hours: float = 24.0
    save_interval_steps: int = 1


class CheckpointManager:
    """
    Manages execution checkpoints for fault recovery.

    Saves execution state after each step (or at configured intervals)
    so that failed executions can be resumed. Supports both file-based
    and Redis-based storage backends.

    Args:
        config: Checkpoint configuration.
    """

    def __init__(self, config: CheckpointConfig | None = None) -> None:
        self._config = config or CheckpointConfig()
        self._checkpoints: dict[str, list[Checkpoint]] = {}

        if self._config.backend == "file":
            Path(self._config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> CheckpointConfig:
        return self._config

    async def save(
        self,
        plan_id: str,
        result: ExecutionResult,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """
        Save a checkpoint of the current execution state.

        Args:
            plan_id: The plan being executed.
            result: Current execution result state.
            metadata: Optional checkpoint metadata.

        Returns:
            The saved Checkpoint.
        """
        checkpoint = Checkpoint(
            plan_id=plan_id,
            execution_id=result.execution_id,
            state=result.model_dump(),
            step_index=len(result.step_results),
            metadata=metadata or {},
        )

        # Track in memory
        if plan_id not in self._checkpoints:
            self._checkpoints[plan_id] = []
        self._checkpoints[plan_id].append(checkpoint)

        # Enforce retention limit
        if len(self._checkpoints[plan_id]) > self._config.max_checkpoints_per_plan:
            self._checkpoints[plan_id] = self._checkpoints[plan_id][
                -self._config.max_checkpoints_per_plan :
            ]

        # Persist
        if self._config.backend == "file":
            await self._save_file(checkpoint)
        elif self._config.backend == "redis":
            await self._save_redis(checkpoint)

        return checkpoint

    async def restore(
        self,
        plan_id: str,
    ) -> ExecutionResult | None:
        """
        Restore the latest checkpoint for a plan.

        Args:
            plan_id: The plan to restore.

        Returns:
            The restored ExecutionResult, or None if no checkpoint exists.
        """
        # Try memory first
        if plan_id in self._checkpoints and self._checkpoints[plan_id]:
            latest = self._checkpoints[plan_id][-1]
            return self._deserialize_result(latest.state)

        # Try file backend
        if self._config.backend == "file":
            return await self._restore_file(plan_id)

        return None

    async def list_checkpoints(self, plan_id: str) -> list[Checkpoint]:
        """List all checkpoints for a plan."""
        return self._checkpoints.get(plan_id, [])

    async def cleanup(self, max_age_hours: float | None = None) -> int:
        """
        Remove old checkpoints.

        Args:
            max_age_hours: Maximum age in hours (uses config default if None).

        Returns:
            Number of checkpoints removed.
        """
        max_age = max_age_hours or self._config.retention_hours
        cutoff = time.time() - (max_age * 3600)
        removed = 0

        for plan_id in list(self._checkpoints):
            before = len(self._checkpoints[plan_id])
            self._checkpoints[plan_id] = [
                c for c in self._checkpoints[plan_id]
                if c.created_at >= cutoff
            ]
            removed += before - len(self._checkpoints[plan_id])

            if not self._checkpoints[plan_id]:
                del self._checkpoints[plan_id]

        # Clean up files
        if self._config.backend == "file":
            removed += await self._cleanup_files(cutoff)

        return removed

    async def _save_file(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to a JSON file."""
        filename = f"{checkpoint.plan_id}_{checkpoint.step_index}_{int(checkpoint.created_at)}.json"
        filepath = Path(self._config.checkpoint_dir) / filename
        data = checkpoint.model_dump()
        filepath.write_text(json.dumps(data, default=str, indent=2))

    async def _restore_file(self, plan_id: str) -> ExecutionResult | None:
        """Restore the latest file checkpoint for a plan."""
        checkpoint_dir = Path(self._config.checkpoint_dir)
        if not checkpoint_dir.exists():
            return None

        matching = sorted(
            checkpoint_dir.glob(f"{plan_id}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not matching:
            return None

        data = json.loads(matching[0].read_text())
        return self._deserialize_result(data.get("state", data))

    async def _save_redis(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to Redis."""
        if self._config.redis_url:
            try:
                import redis.asyncio as aioredis

                r = aioredis.from_url(self._config.redis_url, decode_responses=True)
                key = f"lattice:checkpoint:{checkpoint.plan_id}"
                data = json.dumps(checkpoint.model_dump(), default=str)
                await r.lpush(key, data)
                await r.ltrim(key, 0, self._config.max_checkpoints_per_plan - 1)
                await r.close()
            except ImportError:
                pass

    async def _cleanup_files(self, cutoff: float) -> int:
        """Remove old checkpoint files."""
        checkpoint_dir = Path(self._config.checkpoint_dir)
        if not checkpoint_dir.exists():
            return 0

        removed = 0
        for filepath in checkpoint_dir.glob("*.json"):
            if filepath.stat().st_mtime < cutoff:
                filepath.unlink()
                removed += 1
        return removed

    @staticmethod
    def _deserialize_result(state: dict[str, Any]) -> ExecutionResult | None:
        """Deserialize an ExecutionResult from a state dict."""
        try:
            from lattice.core.executor import ExecutionResult

            return ExecutionResult.model_validate(state)
        except Exception:
            return None
