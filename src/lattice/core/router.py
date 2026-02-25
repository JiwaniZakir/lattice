"""
Learned routing via contextual bandits on task embeddings.

Implements the Epoch-Greedy algorithm from Langford & Zhang (2007) combined
with contextual bandit exploration from Agarwal et al. (2014). The router
maintains per-agent reward estimates conditioned on task embedding vectors
and selects agents using an epsilon-greedy policy with decaying exploration.

The routing model is a lightweight MLP that maps task embeddings to predicted
rewards per agent, trained online as execution feedback arrives.
"""

from __future__ import annotations

import asyncio
import math
import time
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lattice.agents.base import BaseAgent
    from lattice.routing.embedder import TaskEmbedder


class ExplorationStrategy(str, Enum):
    """Exploration strategy for the contextual bandit router."""

    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "ucb"
    THOMPSON = "thompson"


class RoutingDecision(BaseModel):
    """Result of a routing decision with full provenance."""

    agent_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    strategy_used: ExplorationStrategy
    explored: bool
    predicted_rewards: dict[str, float]
    embedding_norm: float
    decision_time_ms: float


class RoutingHistory(BaseModel):
    """A single routing observation for online learning."""

    task_embedding: list[float]
    agent_id: str
    reward: float = Field(ge=0.0, le=1.0)
    timestamp: float = Field(default_factory=time.time)


class RewardPredictor(nn.Module):
    """
    MLP that predicts per-agent rewards from task embeddings.

    Architecture follows the policy network from Agarwal et al. (2014),
    mapping context vectors to action-value estimates. Uses layer
    normalization for training stability under non-stationary reward
    distributions.

    Args:
        embedding_dim: Dimension of the task embedding vector.
        num_agents: Number of agents (arms) to choose from.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, embedding_dim: int, num_agents: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_agents),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict reward for each agent given task embedding(s)."""
        return self.net(x)


class Router:
    """
    Contextual bandit router for multi-agent task assignment.

    Maintains an online-learned reward predictor that maps task embeddings
    to per-agent expected rewards. Supports epsilon-greedy, UCB, and
    Thompson sampling exploration strategies with configurable decay.

    The router implements the following loop:
        1. Receive a task and compute its embedding via TaskEmbedder.
        2. Predict per-agent rewards using the MLP.
        3. Select an agent using the chosen exploration strategy.
        4. After execution, receive reward feedback and update the model.

    Args:
        agents: List of available agents.
        embedding_dim: Dimension of task embedding vectors.
        strategy: Exploration strategy to use.
        epsilon: Initial exploration rate for epsilon-greedy.
        epsilon_decay: Multiplicative decay per update step.
        epsilon_min: Minimum exploration rate.
        learning_rate: Learning rate for the reward predictor.
        device: Torch device for the reward predictor.
    """

    def __init__(
        self,
        agents: Sequence[BaseAgent],
        embedding_dim: int = 384,
        strategy: ExplorationStrategy = ExplorationStrategy.EPSILON_GREEDY,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.01,
        learning_rate: float = 1e-3,
        device: str | torch.device = "cpu",
    ) -> None:
        self._agents = list(agents)
        self._agent_index: dict[str, int] = {
            agent.agent_id: i for i, agent in enumerate(self._agents)
        }
        self._embedding_dim = embedding_dim
        self._strategy = strategy
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        self._device = torch.device(device)

        self._predictor = RewardPredictor(
            embedding_dim=embedding_dim,
            num_agents=len(agents),
        ).to(self._device)

        self._optimizer = torch.optim.AdamW(
            self._predictor.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        # UCB: track counts per agent for confidence bounds
        self._agent_counts = np.zeros(len(agents), dtype=np.float64)
        self._total_rounds = 0

        # Thompson: track per-agent reward statistics (Beta distribution params)
        self._alpha = np.ones(len(agents), dtype=np.float64)
        self._beta_param = np.ones(len(agents), dtype=np.float64)

        # History buffer for batch updates
        self._history: list[RoutingHistory] = []
        self._lock = asyncio.Lock()

    @property
    def agents(self) -> list[BaseAgent]:
        """Currently registered agents."""
        return list(self._agents)

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""
        return self._epsilon

    @property
    def num_agents(self) -> int:
        return len(self._agents)

    def register_agent(self, agent: BaseAgent) -> None:
        """Register a new agent dynamically, expanding the predictor."""
        if agent.agent_id in self._agent_index:
            return
        idx = len(self._agents)
        self._agents.append(agent)
        self._agent_index[agent.agent_id] = idx

        # Expand predictor output layer
        old_linear = self._predictor.net[-1]
        assert isinstance(old_linear, nn.Linear)
        new_linear = nn.Linear(old_linear.in_features, idx + 1).to(self._device)
        with torch.no_grad():
            new_linear.weight[:idx] = old_linear.weight
            new_linear.bias[:idx] = old_linear.bias
            nn.init.xavier_uniform_(new_linear.weight[idx : idx + 1])
            new_linear.bias[idx] = 0.0
        self._predictor.net[-1] = new_linear

        # Expand tracking arrays
        self._agent_counts = np.append(self._agent_counts, 0.0)
        self._alpha = np.append(self._alpha, 1.0)
        self._beta_param = np.append(self._beta_param, 1.0)

        # Re-create optimizer with new parameters
        self._optimizer = torch.optim.AdamW(
            self._predictor.parameters(),
            lr=self._optimizer.defaults["lr"],
            weight_decay=1e-4,
        )

    async def route(
        self,
        task_embedding: np.ndarray | torch.Tensor,
        eligible_agents: Sequence[str] | None = None,
    ) -> RoutingDecision:
        """
        Route a task to the best agent given its embedding.

        Args:
            task_embedding: The embedding vector for the task.
            eligible_agents: Optional subset of agent IDs to consider.

        Returns:
            A RoutingDecision with the selected agent and provenance.
        """
        start = time.perf_counter()

        if isinstance(task_embedding, np.ndarray):
            embedding_tensor = torch.from_numpy(task_embedding).float()
        else:
            embedding_tensor = task_embedding.float()

        if embedding_tensor.dim() == 1:
            embedding_tensor = embedding_tensor.unsqueeze(0)

        embedding_tensor = embedding_tensor.to(self._device)
        embedding_norm = float(embedding_tensor.norm().item())

        # Build mask for eligible agents
        if eligible_agents is not None:
            mask = torch.full((len(self._agents),), float("-inf"), device=self._device)
            for aid in eligible_agents:
                if aid in self._agent_index:
                    mask[self._agent_index[aid]] = 0.0
        else:
            mask = torch.zeros(len(self._agents), device=self._device)

        # Forward pass
        self._predictor.eval()
        with torch.no_grad():
            raw_preds = self._predictor(embedding_tensor).squeeze(0)  # (num_agents,)
            masked_preds = raw_preds + mask

        predicted_rewards = {
            agent.agent_id: float(raw_preds[i].item())
            for i, agent in enumerate(self._agents)
        }

        # Select agent using the exploration strategy
        explored = False
        selected_idx: int

        if self._strategy == ExplorationStrategy.EPSILON_GREEDY:
            selected_idx, explored = self._select_epsilon_greedy(masked_preds)
        elif self._strategy == ExplorationStrategy.UCB:
            selected_idx, explored = self._select_ucb(masked_preds, mask)
        elif self._strategy == ExplorationStrategy.THOMPSON:
            selected_idx, explored = self._select_thompson(masked_preds, mask)
        else:
            selected_idx = int(masked_preds.argmax().item())

        self._agent_counts[selected_idx] += 1
        self._total_rounds += 1

        selected_agent = self._agents[selected_idx]

        # Compute confidence as softmax probability of selected agent
        probs = F.softmax(masked_preds, dim=0)
        confidence = float(probs[selected_idx].item())

        elapsed_ms = (time.perf_counter() - start) * 1000

        return RoutingDecision(
            agent_id=selected_agent.agent_id,
            confidence=min(confidence, 1.0),
            strategy_used=self._strategy,
            explored=explored,
            predicted_rewards=predicted_rewards,
            embedding_norm=embedding_norm,
            decision_time_ms=elapsed_ms,
        )

    def _select_epsilon_greedy(
        self, predictions: torch.Tensor
    ) -> tuple[int, bool]:
        """Epsilon-greedy selection with current epsilon."""
        if np.random.random() < self._epsilon:
            # Explore: uniform over non-masked agents
            valid = (predictions > float("-inf")).nonzero(as_tuple=True)[0]
            idx = valid[torch.randint(len(valid), (1,)).item()].item()
            return int(idx), True
        else:
            return int(predictions.argmax().item()), False

    def _select_ucb(
        self, predictions: torch.Tensor, mask: torch.Tensor
    ) -> tuple[int, bool]:
        """
        Upper Confidence Bound selection.

        UCB1 formula: Q(a) + c * sqrt(ln(t) / N(a))
        """
        c = 2.0
        t = max(self._total_rounds, 1)
        ucb_values = predictions.cpu().numpy().copy()
        for i in range(len(self._agents)):
            if mask[i].item() == float("-inf"):
                ucb_values[i] = float("-inf")
            elif self._agent_counts[i] == 0:
                ucb_values[i] = float("inf")  # explore unvisited
            else:
                bonus = c * math.sqrt(math.log(t) / self._agent_counts[i])
                ucb_values[i] += bonus

        selected = int(np.argmax(ucb_values))
        greedy = int(predictions.argmax().item())
        explored = selected != greedy
        return selected, explored

    def _select_thompson(
        self, predictions: torch.Tensor, mask: torch.Tensor
    ) -> tuple[int, bool]:
        """
        Thompson sampling using Beta posterior.

        Samples from Beta(alpha_a, beta_a) for each agent and selects
        the one with the highest sample.
        """
        samples = np.random.beta(self._alpha, self._beta_param)
        for i in range(len(self._agents)):
            if mask[i].item() == float("-inf"):
                samples[i] = float("-inf")

        selected = int(np.argmax(samples))
        greedy = int(predictions.argmax().item())
        return selected, selected != greedy

    async def update(self, history: RoutingHistory) -> float:
        """
        Update the router with observed reward feedback.

        Performs a single SGD step on the reward predictor and updates
        the Thompson sampling posteriors.

        Args:
            history: Observed routing outcome with reward.

        Returns:
            The training loss for this update step.
        """
        async with self._lock:
            self._history.append(history)

            agent_idx = self._agent_index[history.agent_id]

            # Update Thompson posterior
            self._alpha[agent_idx] += history.reward
            self._beta_param[agent_idx] += 1.0 - history.reward

            # Decay epsilon
            self._epsilon = max(
                self._epsilon * self._epsilon_decay,
                self._epsilon_min,
            )

            # Train predictor
            embedding = (
                torch.tensor(history.task_embedding, dtype=torch.float32)
                .unsqueeze(0)
                .to(self._device)
            )
            target_reward = torch.tensor(
                [history.reward], dtype=torch.float32
            ).to(self._device)

            self._predictor.train()
            self._optimizer.zero_grad()
            preds = self._predictor(embedding).squeeze(0)
            # Only compute loss for the chosen agent
            loss = F.mse_loss(preds[agent_idx].unsqueeze(0), target_reward)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._predictor.parameters(), 1.0)
            self._optimizer.step()

            return float(loss.item())

    async def batch_update(self, histories: Sequence[RoutingHistory]) -> float:
        """
        Batch update the router with multiple observations.

        More efficient than calling update() repeatedly.

        Returns:
            Average training loss over the batch.
        """
        if not histories:
            return 0.0

        async with self._lock:
            embeddings = torch.tensor(
                [h.task_embedding for h in histories],
                dtype=torch.float32,
            ).to(self._device)

            agent_indices = torch.tensor(
                [self._agent_index[h.agent_id] for h in histories],
                dtype=torch.long,
            ).to(self._device)

            rewards = torch.tensor(
                [h.reward for h in histories],
                dtype=torch.float32,
            ).to(self._device)

            # Update Thompson posteriors
            for h in histories:
                idx = self._agent_index[h.agent_id]
                self._alpha[idx] += h.reward
                self._beta_param[idx] += 1.0 - h.reward
                self._history.append(h)

            # Decay epsilon once per batch
            self._epsilon = max(
                self._epsilon * (self._epsilon_decay ** len(histories)),
                self._epsilon_min,
            )

            # Train predictor
            self._predictor.train()
            self._optimizer.zero_grad()
            preds = self._predictor(embeddings)  # (batch, num_agents)

            # Gather predictions for chosen agents
            chosen_preds = preds.gather(1, agent_indices.unsqueeze(1)).squeeze(1)
            loss = F.mse_loss(chosen_preds, rewards)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._predictor.parameters(), 1.0)
            self._optimizer.step()

            return float(loss.item())

    def get_diagnostics(self) -> dict[str, object]:
        """Return diagnostic information about the router state."""
        return {
            "epsilon": self._epsilon,
            "total_rounds": self._total_rounds,
            "agent_counts": {
                agent.agent_id: int(self._agent_counts[i])
                for i, agent in enumerate(self._agents)
            },
            "thompson_alpha": {
                agent.agent_id: float(self._alpha[i])
                for i, agent in enumerate(self._agents)
            },
            "thompson_beta": {
                agent.agent_id: float(self._beta_param[i])
                for i, agent in enumerate(self._agents)
            },
            "history_size": len(self._history),
            "strategy": self._strategy.value,
        }
