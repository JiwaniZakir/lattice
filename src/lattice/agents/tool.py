"""
Tool-using agent that can invoke external tools during task execution.

Implements the ReAct pattern (Yao et al., 2023) for tool use: the agent
iterates through thought-action-observation loops, selecting and calling
tools at each step until the task is complete or the step limit is reached.

Tools are registered as typed callables with JSON-schema parameter
descriptions, enabling the LLM to generate valid tool calls.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Coroutine
from typing import Any

from pydantic import BaseModel, Field

from lattice.agents.base import AgentCapabilities, AgentResult, AgentStatus, BaseAgent


class ToolDefinition(BaseModel):
    """Definition of a tool available to the agent."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    returns: str = "str"
    requires_confirmation: bool = False


class ToolCall(BaseModel):
    """A recorded tool invocation."""

    tool_name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0


class ToolAgent(BaseAgent):
    """
    Agent that uses tools via the ReAct pattern.

    The agent maintains a registry of tools (async callables) and uses
    an LLM to decide which tool to call at each step. The ReAct loop:

        1. Thought: reason about the current state
        2. Action: select a tool and arguments
        3. Observation: execute the tool and observe the result
        4. Repeat until done or step limit reached

    Args:
        agent_id: Unique identifier.
        name: Human-readable name.
        description: Description of the agent.
        llm_provider: LLM provider for reasoning.
        model: Model to use for reasoning.
        max_steps: Maximum ReAct iterations.
        tools: Optional initial tool registry.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        name: str = "ToolAgent",
        description: str = "An agent that uses tools to complete tasks",
        llm_provider: Any = None,
        model: str = "gpt-4o",
        max_steps: int = 10,
        tools: dict[str, tuple[ToolDefinition, Callable[..., Coroutine[Any, Any, Any]]]] | None = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            name=name,
            description=description,
            capabilities=AgentCapabilities(
                can_use_tools=True,
                can_stream=True,
                supported_task_types=["tool_use", "code", "research", "general"],
            ),
        )
        self._llm = llm_provider
        self._model = model
        self._max_steps = max_steps
        self._tools: dict[str, tuple[ToolDefinition, Callable[..., Coroutine[Any, Any, Any]]]] = (
            tools or {}
        )
        self._call_history: list[ToolCall] = []

    def register_tool(
        self,
        name: str,
        description: str,
        fn: Callable[..., Coroutine[Any, Any, Any]],
        parameters: dict[str, Any] | None = None,
        requires_confirmation: bool = False,
    ) -> None:
        """
        Register a tool for use during execution.

        Args:
            name: Tool name (used in LLM tool calls).
            description: Description of what the tool does.
            fn: Async callable that implements the tool.
            parameters: JSON-schema-style parameter descriptions.
            requires_confirmation: Whether this tool needs human approval.
        """
        defn = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or {},
            requires_confirmation=requires_confirmation,
        )
        self._tools[name] = (defn, fn)

    def get_tool_definitions(self) -> list[ToolDefinition]:
        """Return all registered tool definitions."""
        return [defn for defn, _ in self._tools.values()]

    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Execute a task using the ReAct tool-use loop.

        If no LLM provider is configured, falls back to a simple
        keyword-based tool selection heuristic.

        Args:
            task: The task to execute.
            context: Optional context from prior steps.

        Returns:
            AgentResult with the output.
        """
        start = time.perf_counter()
        self._status = AgentStatus.BUSY
        total_tokens = 0
        total_cost = 0.0
        tool_calls: list[ToolCall] = []

        try:
            if self._llm:
                output = await self._react_loop(task, context, tool_calls)
            else:
                output = await self._heuristic_execute(task, context, tool_calls)

            latency = (time.perf_counter() - start) * 1000
            result = AgentResult(
                agent_id=self._agent_id,
                task=task,
                output=output,
                success=True,
                tokens_used=total_tokens,
                cost_usd=total_cost,
                latency_ms=latency,
                metadata={
                    "tool_calls": [tc.model_dump() for tc in tool_calls],
                    "num_steps": len(tool_calls),
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

    async def _react_loop(
        self,
        task: str,
        context: dict[str, Any] | None,
        tool_calls: list[ToolCall],
    ) -> str:
        """Full ReAct loop using the LLM provider."""
        tool_defs = self.get_tool_definitions()
        tools_desc = "\n".join(
            f"- {t.name}: {t.description}" for t in tool_defs
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a tool-using agent. Available tools:\n"
                    f"{tools_desc}\n\n"
                    "Respond with JSON: {\"thought\": \"...\", \"tool\": \"name\", "
                    "\"args\": {...}} or {\"thought\": \"...\", \"answer\": \"...\"} "
                    "when done."
                ),
            },
            {"role": "user", "content": task},
        ]

        if context:
            messages.append({
                "role": "assistant",
                "content": f"Context from prior steps: {json.dumps(context, default=str)}",
            })

        for step in range(self._max_steps):
            response = await self._llm.complete(
                messages=messages,
                model=self._model,
                temperature=0.1,
            )

            try:
                parsed = json.loads(response.content)
            except json.JSONDecodeError:
                return response.content

            if "answer" in parsed:
                return parsed["answer"]

            if "tool" in parsed:
                tool_name = parsed["tool"]
                args = parsed.get("args", {})
                tc = await self._call_tool(tool_name, args)
                tool_calls.append(tc)

                observation = tc.result if tc.error is None else f"Error: {tc.error}"
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(parsed),
                })
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}",
                })

        return "Max steps reached without completion"

    async def _heuristic_execute(
        self,
        task: str,
        context: dict[str, Any] | None,
        tool_calls: list[ToolCall],
    ) -> Any:
        """Simple keyword-based tool selection fallback."""
        task_lower = task.lower()
        for tool_name, (defn, fn) in self._tools.items():
            if tool_name.lower() in task_lower or defn.description.lower() in task_lower:
                tc = await self._call_tool(tool_name, {"task": task})
                tool_calls.append(tc)
                if tc.error is None:
                    return tc.result

        # No matching tool; return the task as a pass-through
        return f"No suitable tool found for: {task}"

    async def _call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ToolCall:
        """Call a registered tool and record the result."""
        if tool_name not in self._tools:
            return ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                error=f"Unknown tool: {tool_name}",
            )

        _, fn = self._tools[tool_name]
        start = time.perf_counter()
        try:
            result = await fn(**arguments)
            latency = (time.perf_counter() - start) * 1000
            call = ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            call = ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                error=str(e),
                latency_ms=latency,
            )

        self._call_history.append(call)
        return call
