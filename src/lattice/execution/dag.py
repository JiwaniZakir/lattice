"""
DAG construction and topological execution engine.

Builds a directed acyclic graph from sub-goals and their dependency
edges, validates the structure, and provides topological iteration
for the executor. Supports:

    - Topological sort with cycle detection
    - Critical path analysis
    - Parallel execution wave computation
    - Subgraph extraction for partial re-execution
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from lattice.core.planner import SubGoal


class DAGNode(BaseModel):
    """A node in the execution DAG."""

    node_id: str
    label: str = ""
    agent_id: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DAGEdge(BaseModel):
    """A directed edge in the DAG."""

    source: str
    target: str
    label: str = ""


class DAGInfo(BaseModel):
    """Analysis information about a DAG."""

    num_nodes: int
    num_edges: int
    depth: int
    width: int
    critical_path: list[str]
    execution_waves: list[list[str]]
    is_valid: bool
    cycle_nodes: list[str] = Field(default_factory=list)


class DAGBuilder:
    """
    Builds and analyzes execution DAGs from sub-goals.

    The builder constructs a graph structure, validates it (acyclicity,
    connectivity), and computes execution metadata like the critical
    path and parallel execution waves.

    Example::

        builder = DAGBuilder()
        for goal in plan.sub_goals:
            builder.add_node(goal.goal_id, goal.description, goal.assigned_agent)
            for dep in goal.dependencies:
                builder.add_edge(dep, goal.goal_id)
        dag = builder.build()
    """

    def __init__(self) -> None:
        self._nodes: dict[str, DAGNode] = {}
        self._edges: list[DAGEdge] = []
        self._adjacency: dict[str, list[str]] = defaultdict(list)
        self._reverse_adj: dict[str, list[str]] = defaultdict(list)

    def add_node(
        self,
        node_id: str,
        label: str = "",
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DAGNode:
        """Add a node to the DAG."""
        node = DAGNode(
            node_id=node_id,
            label=label,
            agent_id=agent_id,
            metadata=metadata or {},
        )
        self._nodes[node_id] = node
        return node

    def add_edge(self, source: str, target: str, label: str = "") -> DAGEdge:
        """Add a directed edge from source to target."""
        edge = DAGEdge(source=source, target=target, label=label)
        self._edges.append(edge)
        self._adjacency[source].append(target)
        self._reverse_adj[target].append(source)

        # Update node dependencies
        if target in self._nodes:
            if source not in self._nodes[target].dependencies:
                self._nodes[target].dependencies.append(source)

        return edge

    @classmethod
    def from_sub_goals(cls, sub_goals: list[SubGoal]) -> DAGBuilder:
        """Create a DAGBuilder from a list of sub-goals."""
        builder = cls()
        for goal in sub_goals:
            builder.add_node(
                goal.goal_id,
                label=goal.description,
                agent_id=goal.assigned_agent,
            )
        for goal in sub_goals:
            for dep in goal.dependencies:
                if dep in builder._nodes:
                    builder.add_edge(dep, goal.goal_id)
        return builder

    def build(self) -> DAGInfo:
        """
        Validate the DAG and compute execution metadata.

        Returns:
            DAGInfo with analysis results.
        """
        is_valid, cycle_nodes = self._check_acyclic()
        topo_order = self._topological_sort() if is_valid else []
        waves = self._compute_waves() if is_valid else []
        critical = self._critical_path() if is_valid else []
        depth = len(waves)
        width = max((len(w) for w in waves), default=0)

        return DAGInfo(
            num_nodes=len(self._nodes),
            num_edges=len(self._edges),
            depth=depth,
            width=width,
            critical_path=critical,
            execution_waves=waves,
            is_valid=is_valid,
            cycle_nodes=cycle_nodes,
        )

    def _check_acyclic(self) -> tuple[bool, list[str]]:
        """
        Check if the graph is acyclic using Kahn's algorithm.

        Returns:
            Tuple of (is_acyclic, list of nodes in cycle).
        """
        in_degree: dict[str, int] = {n: 0 for n in self._nodes}
        for edge in self._edges:
            if edge.target in in_degree:
                in_degree[edge.target] += 1

        queue: deque[str] = deque(
            n for n, d in in_degree.items() if d == 0
        )
        visited = 0

        while queue:
            node = queue.popleft()
            visited += 1
            for neighbor in self._adjacency.get(node, []):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        if visited == len(self._nodes):
            return True, []

        # Find nodes in cycle (those with remaining in-degree > 0)
        cycle_nodes = [n for n, d in in_degree.items() if d > 0]
        return False, cycle_nodes

    def _topological_sort(self) -> list[str]:
        """Compute topological ordering using Kahn's algorithm."""
        in_degree: dict[str, int] = {n: 0 for n in self._nodes}
        for edge in self._edges:
            if edge.target in in_degree:
                in_degree[edge.target] += 1

        queue: deque[str] = deque(
            n for n, d in in_degree.items() if d == 0
        )
        order: list[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in self._adjacency.get(node, []):
                if neighbor in in_degree:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        return order

    def _compute_waves(self) -> list[list[str]]:
        """
        Compute execution waves (parallel layers).

        Each wave contains nodes whose dependencies are all in previous waves.
        """
        in_degree: dict[str, int] = {n: 0 for n in self._nodes}
        for edge in self._edges:
            if edge.target in in_degree:
                in_degree[edge.target] += 1

        current_wave = [n for n, d in in_degree.items() if d == 0]
        waves: list[list[str]] = []

        while current_wave:
            waves.append(current_wave)
            next_wave: list[str] = []
            for node in current_wave:
                for neighbor in self._adjacency.get(node, []):
                    if neighbor in in_degree:
                        in_degree[neighbor] -= 1
                        if in_degree[neighbor] == 0:
                            next_wave.append(neighbor)
            current_wave = next_wave

        return waves

    def _critical_path(self) -> list[str]:
        """
        Find the critical path (longest path) in the DAG.

        Uses dynamic programming on the topological order.
        """
        topo = self._topological_sort()
        if not topo:
            return []

        # Longest path from each node
        dist: dict[str, int] = {n: 0 for n in self._nodes}
        pred: dict[str, str | None] = {n: None for n in self._nodes}

        for node in topo:
            for neighbor in self._adjacency.get(node, []):
                if neighbor in dist and dist[node] + 1 > dist[neighbor]:
                    dist[neighbor] = dist[node] + 1
                    pred[neighbor] = node

        # Find the node with maximum distance
        if not dist:
            return []
        end = max(dist, key=lambda n: dist[n])

        # Trace back
        path: list[str] = []
        current: str | None = end
        while current is not None:
            path.append(current)
            current = pred.get(current)
        path.reverse()
        return path

    def get_subgraph(self, node_ids: set[str]) -> DAGBuilder:
        """Extract a subgraph containing only the specified nodes."""
        sub = DAGBuilder()
        for nid in node_ids:
            if nid in self._nodes:
                node = self._nodes[nid]
                sub.add_node(
                    nid,
                    label=node.label,
                    agent_id=node.agent_id,
                    metadata=node.metadata,
                )
        for edge in self._edges:
            if edge.source in node_ids and edge.target in node_ids:
                sub.add_edge(edge.source, edge.target, edge.label)
        return sub
