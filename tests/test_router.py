"""Tests for the contextual bandit router."""

from __future__ import annotations

import numpy as np
import pytest

from lattice.agents.base import AgentResult, BaseAgent
from lattice.core.router import (
    ExplorationStrategy,
    Router,
    RoutingHistory,
)


# -- Fixtures ----------------------------------------------------------------

class MockAgent(BaseAgent):
    """Simple mock agent for testing."""

    async def execute(self, task: str, context: dict | None = None) -> AgentResult:
        return AgentResult(agent_id=self.agent_id, task=task, output="mock")


@pytest.fixture
def agents() -> list[MockAgent]:
    return [
        MockAgent(agent_id="agent_a", name="Agent A"),
        MockAgent(agent_id="agent_b", name="Agent B"),
        MockAgent(agent_id="agent_c", name="Agent C"),
    ]


@pytest.fixture
def router(agents: list[MockAgent]) -> Router:
    return Router(
        agents=agents,
        embedding_dim=64,
        strategy=ExplorationStrategy.EPSILON_GREEDY,
        epsilon=0.1,
    )


# -- Tests -------------------------------------------------------------------

class TestRouterInit:
    def test_router_has_correct_agent_count(self, router: Router) -> None:
        assert router.num_agents == 3

    def test_router_agents_accessible(self, router: Router) -> None:
        ids = {a.agent_id for a in router.agents}
        assert ids == {"agent_a", "agent_b", "agent_c"}

    def test_initial_epsilon(self, router: Router) -> None:
        assert router.epsilon == 0.1


class TestRouterRouting:
    @pytest.mark.asyncio
    async def test_route_returns_valid_decision(self, router: Router) -> None:
        embedding = np.random.randn(64).astype(np.float32)
        decision = await router.route(embedding)

        assert decision.agent_id in {"agent_a", "agent_b", "agent_c"}
        assert 0.0 <= decision.confidence <= 1.0
        assert decision.strategy_used == ExplorationStrategy.EPSILON_GREEDY
        assert decision.decision_time_ms > 0
        assert decision.embedding_norm > 0

    @pytest.mark.asyncio
    async def test_route_respects_eligible_agents(self, router: Router) -> None:
        embedding = np.random.randn(64).astype(np.float32)
        decision = await router.route(embedding, eligible_agents=["agent_b"])

        assert decision.agent_id == "agent_b"

    @pytest.mark.asyncio
    async def test_route_with_torch_tensor(self, router: Router) -> None:
        import torch
        embedding = torch.randn(64)
        decision = await router.route(embedding)
        assert decision.agent_id in {"agent_a", "agent_b", "agent_c"}

    @pytest.mark.asyncio
    async def test_predicted_rewards_for_all_agents(self, router: Router) -> None:
        embedding = np.random.randn(64).astype(np.float32)
        decision = await router.route(embedding)
        assert set(decision.predicted_rewards.keys()) == {"agent_a", "agent_b", "agent_c"}


class TestRouterUpdate:
    @pytest.mark.asyncio
    async def test_update_decreases_epsilon(self, router: Router) -> None:
        initial_eps = router.epsilon
        embedding = np.random.randn(64).tolist()

        loss = await router.update(RoutingHistory(
            task_embedding=embedding,
            agent_id="agent_a",
            reward=0.8,
        ))

        assert router.epsilon < initial_eps
        assert loss >= 0.0

    @pytest.mark.asyncio
    async def test_batch_update(self, router: Router) -> None:
        histories = [
            RoutingHistory(
                task_embedding=np.random.randn(64).tolist(),
                agent_id="agent_a",
                reward=0.9,
            )
            for _ in range(5)
        ]

        loss = await router.batch_update(histories)
        assert loss >= 0.0

    @pytest.mark.asyncio
    async def test_empty_batch_update(self, router: Router) -> None:
        loss = await router.batch_update([])
        assert loss == 0.0


class TestRouterDynamicAgents:
    @pytest.mark.asyncio
    async def test_register_new_agent(self, router: Router) -> None:
        new_agent = MockAgent(agent_id="agent_d", name="Agent D")
        router.register_agent(new_agent)

        assert router.num_agents == 4
        embedding = np.random.randn(64).astype(np.float32)
        decision = await router.route(embedding, eligible_agents=["agent_d"])
        assert decision.agent_id == "agent_d"

    @pytest.mark.asyncio
    async def test_register_duplicate_agent(self, router: Router) -> None:
        dup = MockAgent(agent_id="agent_a", name="Duplicate")
        router.register_agent(dup)
        assert router.num_agents == 3  # No change


class TestRouterStrategies:
    @pytest.mark.asyncio
    async def test_ucb_strategy(self, agents: list[MockAgent]) -> None:
        router = Router(
            agents=agents,
            embedding_dim=64,
            strategy=ExplorationStrategy.UCB,
        )
        embedding = np.random.randn(64).astype(np.float32)
        decision = await router.route(embedding)
        assert decision.strategy_used == ExplorationStrategy.UCB

    @pytest.mark.asyncio
    async def test_thompson_strategy(self, agents: list[MockAgent]) -> None:
        router = Router(
            agents=agents,
            embedding_dim=64,
            strategy=ExplorationStrategy.THOMPSON,
        )
        embedding = np.random.randn(64).astype(np.float32)
        decision = await router.route(embedding)
        assert decision.strategy_used == ExplorationStrategy.THOMPSON


class TestRouterDiagnostics:
    @pytest.mark.asyncio
    async def test_diagnostics_structure(self, router: Router) -> None:
        diag = router.get_diagnostics()
        assert "epsilon" in diag
        assert "total_rounds" in diag
        assert "agent_counts" in diag
        assert "strategy" in diag
        assert diag["strategy"] == "epsilon_greedy"
