# Lattice

**Adaptive Multi-Agent Orchestration Framework**

Route, plan, verify, and execute multi-agent workflows with learned routing, formal safety proofs, and full observability.

---

## Overview

Lattice is a Python framework for building **multi-agent systems with formal safety guarantees**. It combines:

- A **contextual bandit router** that learns optimal agent assignment from execution feedback
- A **hybrid ReAct + Plan-and-Solve planner** for task decomposition
- A **Z3-powered verifier** that proves DAG acyclicity, budget feasibility, and capability constraints before any code executes

The result is a production-grade orchestration layer where agents are routed intelligently, plans are verified mathematically, and every step is traced end-to-end via OpenTelemetry.

---

## Installation

```bash
pip install lattice
```

Or install from source with dev dependencies:

```bash
git clone https://github.com/JiwaniZakir/lattice.git
cd lattice
pip install -e ".[dev]"
```

---

## Quick Start

```python
import asyncio
import numpy as np
from lattice import Router, Planner, Verifier, Executor
from lattice.agents.base import AgentResult, BaseAgent

# Define a custom agent
class MyAgent(BaseAgent):
    async def execute(self, task, context=None):
        return AgentResult(
            agent_id=self.agent_id,
            task=task,
            output=f"Completed: {task}",
        )

async def main():
    agent = MyAgent(agent_id="my_agent", name="My Agent")

    # 1. Route -- contextual bandit selects the best agent
    router = Router(agents=[agent], embedding_dim=384)
    embedding = np.random.randn(384).astype(np.float32)
    decision = await router.route(embedding)

    # 2. Plan -- decompose task into sub-goals
    planner = Planner()
    plan = await planner.plan(
        task="Analyze the quarterly report",
        available_agents=[decision.agent_id],
    )

    # 3. Verify -- Z3 proves the plan is safe
    verifier = Verifier()
    check = await verifier.verify_plan(plan)
    assert check.is_safe

    # 4. Execute -- DAG engine runs sub-goals with parallelism
    executor = Executor(agents={"my_agent": agent})
    result = await executor.execute(plan)
    print(result.status, result.total_cost_usd)

asyncio.run(main())
```

---

## Features

| Feature | Description |
|---|---|
| **Learned Routing** | Contextual bandit router (epsilon-greedy, UCB, Thompson sampling) that learns optimal agent assignment from task embeddings and execution feedback |
| **Adaptive Planning** | Hybrid ReAct + Plan-and-Solve planner with Voyager-style skill caching |
| **Formal Verification** | Z3-based safety verification: DAG acyclicity, budget feasibility, capability matching, and custom policies |
| **DAG Execution** | Topological execution engine with dependency tracking, parallel dispatch, retry with exponential backoff, and timeout |
| **Scoped Memory** | Hierarchical shared memory with namespace inheritance, TTL expiration, semantic search, LRU eviction, and optional Redis persistence |
| **Token Streaming** | Async token-level streaming with backpressure and multi-consumer fan-out |
| **Checkpointing** | Fault recovery by resuming failed plans from the last successful step |
| **Cost Attribution** | Per-agent, per-model cost tracking via LiteLLM (100+ models) |
| **OpenTelemetry** | Full distributed tracing exportable to Jaeger, Honeycomb, or any OTLP collector |
| **Constitutional Critic** | Evaluation agent that scores outputs against configurable principles |
| **Human-in-the-Loop** | Queue-based human approval agent with timeout and audit logging |

---

## Project Structure

```
src/lattice/
├── core/               # Router, Planner, Executor, Verifier, Memory
├── agents/             # BaseAgent, ToolAgent, CriticAgent, HumanAgent
├── routing/            # Classifier, embedder, feedback processing
├── verification/       # Z3 invariants, solver wrapper, policies
├── execution/          # DAG builder, streaming, checkpointing
├── observability/      # OpenTelemetry tracing, metrics, logging
└── integrations/       # OpenAI, Anthropic, LiteLLM providers
```

---

## Links

- [GitHub Repository](https://github.com/JiwaniZakir/lattice)
- [Issue Tracker](https://github.com/JiwaniZakir/lattice/issues)
- [License: MIT](https://github.com/JiwaniZakir/lattice/blob/main/LICENSE)
