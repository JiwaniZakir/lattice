"""
Human-in-the-loop agent for tasks requiring human judgment.

Provides a mechanism for injecting human decisions into automated
workflows. The agent queues tasks for human review and blocks
(with timeout) until a response is provided. This is essential for:

    - Approving high-stakes actions
    - Providing subjective judgments
    - Handling edge cases the AI cannot resolve
    - Compliance and audit requirements

Supports both synchronous (blocking) and callback-based patterns.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from pydantic import BaseModel, Field

from lattice.agents.base import AgentCapabilities, AgentResult, AgentStatus, BaseAgent


class HumanRequest(BaseModel):
    """A request queued for human review."""

    request_id: str
    task: str
    context: dict[str, Any] = Field(default_factory=dict)
    options: list[str] | None = None
    created_at: float = Field(default_factory=time.time)
    timeout_s: float = 300.0


class HumanResponse(BaseModel):
    """A human's response to a queued request."""

    request_id: str
    response: str
    approved: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    responded_at: float = Field(default_factory=time.time)


class HumanAgent(BaseAgent):
    """
    Agent that routes tasks to a human for review/response.

    Tasks are placed in a queue and the agent waits (with timeout)
    for a human to provide a response via the `respond` method.
    This enables human-in-the-loop workflows where certain decisions
    require human judgment or approval.

    For automated testing, responses can be pre-loaded via
    `preload_response`.

    Args:
        agent_id: Unique identifier.
        name: Human-readable name.
        default_timeout_s: Default timeout for human responses.
        auto_approve: If True, automatically approves all requests (testing mode).
    """

    def __init__(
        self,
        agent_id: str | None = None,
        name: str = "HumanAgent",
        default_timeout_s: float = 300.0,
        auto_approve: bool = False,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            name=name,
            description="Routes tasks to a human for review",
            capabilities=AgentCapabilities(
                requires_human_approval=True,
                supported_task_types=["review", "approval", "judgment"],
            ),
        )
        self._default_timeout = default_timeout_s
        self._auto_approve = auto_approve
        self._pending: dict[str, asyncio.Future[HumanResponse]] = {}
        self._preloaded: dict[str, HumanResponse] = {}
        self._request_log: list[HumanRequest] = []

    @property
    def pending_count(self) -> int:
        """Number of requests awaiting human response."""
        return len(self._pending)

    def preload_response(self, request_id: str, response: HumanResponse) -> None:
        """Pre-load a response for automated testing."""
        self._preloaded[request_id] = response

    def preload_auto_response(self, response_text: str = "Approved") -> None:
        """Set a default auto-response for all requests (testing mode)."""
        self._auto_approve = True
        self._default_response_text = response_text

    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Submit a task for human review and wait for response.

        Args:
            task: The task requiring human input.
            context: Optional context for the human reviewer.

        Returns:
            AgentResult with the human's response.
        """
        import uuid

        start = time.perf_counter()
        self._status = AgentStatus.BUSY
        request_id = uuid.uuid4().hex[:12]

        request = HumanRequest(
            request_id=request_id,
            task=task,
            context=context or {},
            timeout_s=self._default_timeout,
        )
        self._request_log.append(request)

        try:
            # Check for preloaded response
            if request_id in self._preloaded:
                response = self._preloaded.pop(request_id)
            elif self._auto_approve:
                response_text = getattr(self, "_default_response_text", "Approved")
                response = HumanResponse(
                    request_id=request_id,
                    response=response_text,
                    approved=True,
                )
            else:
                # Wait for human response
                loop = asyncio.get_event_loop()
                future: asyncio.Future[HumanResponse] = loop.create_future()
                self._pending[request_id] = future

                try:
                    response = await asyncio.wait_for(
                        future, timeout=self._default_timeout
                    )
                except asyncio.TimeoutError:
                    self._pending.pop(request_id, None)
                    latency = (time.perf_counter() - start) * 1000
                    result = AgentResult(
                        agent_id=self._agent_id,
                        task=task,
                        output=None,
                        success=False,
                        error=f"Human response timed out after {self._default_timeout}s",
                        latency_ms=latency,
                    )
                    self._status = AgentStatus.IDLE
                    self._record_execution(result)
                    return result
                finally:
                    self._pending.pop(request_id, None)

            latency = (time.perf_counter() - start) * 1000
            result = AgentResult(
                agent_id=self._agent_id,
                task=task,
                output=response.response,
                success=response.approved,
                latency_ms=latency,
                metadata={
                    "request_id": request_id,
                    "approved": response.approved,
                    "response_metadata": response.metadata,
                },
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            result = AgentResult(
                agent_id=self._agent_id,
                task=task,
                output=None,
                success=False,
                error=str(e),
                latency_ms=latency,
            )
        finally:
            self._status = AgentStatus.IDLE
            self._record_execution(result)

        return result

    def respond(self, request_id: str, response: HumanResponse) -> bool:
        """
        Provide a human response to a pending request.

        Args:
            request_id: The request to respond to.
            response: The human's response.

        Returns:
            True if the response was delivered, False if no pending request.
        """
        future = self._pending.get(request_id)
        if future and not future.done():
            future.set_result(response)
            return True
        return False

    def get_pending_requests(self) -> list[HumanRequest]:
        """Return all pending requests awaiting human response."""
        return [
            r for r in self._request_log
            if r.request_id in self._pending
        ]
