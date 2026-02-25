"""
Abstract agent interface with protocol definitions.

Defines the base contract for all agents in the Lattice framework.
Agents are the fundamental execution units -- they receive tasks
from the executor and produce results. Each agent has:

    - A unique identifier
    - A set of declared capabilities
    - An async execute method
    - Optional streaming support
    - Cost tracking
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """Current status of an agent."""

    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    DISABLED = "disabled"


class AgentResult(BaseModel):
    """Result returned by an agent after executing a task."""

    agent_id: str
    task: str
    output: Any
    success: bool = True
    error: str | None = None
    tokens_used: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class AgentCapabilities(BaseModel):
    """Declared capabilities of an agent."""

    can_stream: bool = False
    can_use_tools: bool = False
    supported_models: list[str] = Field(default_factory=list)
    supported_task_types: list[str] = Field(default_factory=list)
    max_context_tokens: int = 128_000
    requires_human_approval: bool = False


@runtime_checkable
class StreamingAgent(Protocol):
    """Protocol for agents that support token-level streaming."""

    async def execute_streaming(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a task with streaming output."""
        ...


class BaseAgent(ABC):
    """
    Abstract base class for all Lattice agents.

    Subclasses must implement the `execute` method. Agents are identified
    by a unique agent_id and declare their capabilities upfront so the
    router and verifier can make informed decisions.

    Args:
        agent_id: Unique identifier for this agent.
        name: Human-readable name.
        description: Description of the agent's purpose.
        capabilities: Declared capabilities.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        name: str = "Agent",
        description: str = "",
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        self._agent_id = agent_id or uuid.uuid4().hex[:12]
        self._name = name
        self._description = description
        self._capabilities = capabilities or AgentCapabilities()
        self._status = AgentStatus.IDLE
        self._total_tasks = 0
        self._total_tokens = 0
        self._total_cost = 0.0

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def capabilities(self) -> AgentCapabilities:
        return self._capabilities

    @property
    def status(self) -> AgentStatus:
        return self._status

    @abstractmethod
    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Execute a task and return a result.

        Args:
            task: The task description to execute.
            context: Optional context from prior execution steps.

        Returns:
            AgentResult with the output and metadata.
        """
        ...

    async def health_check(self) -> bool:
        """Check if this agent is operational."""
        return self._status != AgentStatus.DISABLED

    def get_stats(self) -> dict[str, Any]:
        """Return execution statistics for this agent."""
        return {
            "agent_id": self._agent_id,
            "name": self._name,
            "status": self._status.value,
            "total_tasks": self._total_tasks,
            "total_tokens": self._total_tokens,
            "total_cost_usd": self._total_cost,
        }

    def _record_execution(self, result: AgentResult) -> None:
        """Update internal counters after execution."""
        self._total_tasks += 1
        self._total_tokens += result.tokens_used
        self._total_cost += result.cost_usd
