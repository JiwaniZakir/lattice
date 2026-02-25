"""
Lattice -- Adaptive Multi-Agent Orchestration Framework.

Hierarchical multi-agent orchestration with learned routing,
adaptive planning, and formal workflow verification.

Papers implemented:
    - Voyager (Wang et al., 2023)
    - Plan-and-Solve (Wang et al., 2023)
    - ReAct (Yao et al., 2023)
    - Contextual Bandits (Agarwal et al., 2014)
    - Constitutional AI (Bai et al., 2022)
"""

from lattice.core.executor import Executor
from lattice.core.memory import ScopedMemory
from lattice.core.planner import Planner
from lattice.core.router import Router
from lattice.core.verifier import Verifier

__version__ = "0.1.0"

__all__ = [
    "Executor",
    "Planner",
    "Router",
    "ScopedMemory",
    "Verifier",
    "__version__",
]
