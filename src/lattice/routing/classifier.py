"""
Task classifier for routing decisions.

Classifies incoming tasks into categories that the router uses to
make agent selection decisions. The classifier operates on task
embeddings and supports both rule-based and learned classification.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field


class TaskCategory(str, Enum):
    """Standard task categories for routing."""

    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    RESEARCH = "research"
    WRITING = "writing"
    ANALYSIS = "analysis"
    MATH = "math"
    CONVERSATION = "conversation"
    TOOL_USE = "tool_use"
    PLANNING = "planning"
    EVALUATION = "evaluation"
    OTHER = "other"


class ClassificationResult(BaseModel):
    """Result of classifying a task."""

    category: TaskCategory
    confidence: float = Field(ge=0.0, le=1.0)
    all_scores: dict[str, float] = Field(default_factory=dict)
    features_used: list[str] = Field(default_factory=list)


class ClassifierHead(nn.Module):
    """
    Lightweight classification head on top of task embeddings.

    Maps embedding vectors to task category probabilities using
    a single hidden layer with dropout.

    Args:
        embedding_dim: Dimension of input embeddings.
        num_categories: Number of output categories.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(
        self, embedding_dim: int = 384, num_categories: int = 11, hidden_dim: int = 64
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_categories),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Keyword-based rules for classification when no model is available.
_KEYWORD_RULES: dict[TaskCategory, list[str]] = {
    TaskCategory.CODE_GENERATION: [
        "write code", "implement", "create a function", "program",
        "script", "generate code", "coding",
    ],
    TaskCategory.CODE_REVIEW: [
        "review code", "code review", "find bugs", "refactor",
        "improve code", "lint",
    ],
    TaskCategory.RESEARCH: [
        "research", "find information", "look up", "search for",
        "investigate", "survey", "literature",
    ],
    TaskCategory.WRITING: [
        "write", "draft", "compose", "essay", "article",
        "blog post", "email", "letter",
    ],
    TaskCategory.ANALYSIS: [
        "analyze", "analysis", "evaluate data", "statistics",
        "trends", "compare", "assess",
    ],
    TaskCategory.MATH: [
        "calculate", "solve", "equation", "math", "proof",
        "formula", "compute", "derivative", "integral",
    ],
    TaskCategory.TOOL_USE: [
        "use tool", "call api", "run command", "execute",
        "fetch", "query database",
    ],
    TaskCategory.PLANNING: [
        "plan", "organize", "schedule", "strategy",
        "roadmap", "outline", "prioritize",
    ],
    TaskCategory.EVALUATION: [
        "evaluate", "grade", "score", "rate", "assess quality",
        "benchmark", "test",
    ],
}


class TaskClassifier:
    """
    Task classifier supporting both rule-based and learned classification.

    When an embedding model is available, uses a learned classifier head.
    Otherwise falls back to keyword matching rules.

    Args:
        embedding_dim: Dimension of task embeddings.
        device: Torch device for the classifier head.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        device: str | torch.device = "cpu",
    ) -> None:
        self._device = torch.device(device)
        categories = list(TaskCategory)
        self._categories = categories
        self._category_to_idx = {c: i for i, c in enumerate(categories)}
        self._classifier = ClassifierHead(
            embedding_dim=embedding_dim,
            num_categories=len(categories),
        ).to(self._device)

    def classify_text(self, task: str) -> ClassificationResult:
        """
        Classify a task using keyword rules.

        Args:
            task: Task description text.

        Returns:
            ClassificationResult with category and confidence.
        """
        task_lower = task.lower()
        scores: dict[str, float] = {}

        for category, keywords in _KEYWORD_RULES.items():
            match_count = sum(1 for kw in keywords if kw in task_lower)
            scores[category.value] = match_count / max(len(keywords), 1)

        # Default to OTHER with low confidence
        scores.setdefault(TaskCategory.OTHER.value, 0.1)

        best_category = max(scores, key=lambda k: scores[k])
        best_score = scores[best_category]

        if best_score == 0:
            best_category = TaskCategory.OTHER.value
            best_score = 0.5

        return ClassificationResult(
            category=TaskCategory(best_category),
            confidence=min(best_score, 1.0),
            all_scores=scores,
            features_used=["keyword_rules"],
        )

    def classify_embedding(
        self, embedding: np.ndarray | torch.Tensor
    ) -> ClassificationResult:
        """
        Classify a task using its embedding vector.

        Args:
            embedding: Task embedding vector.

        Returns:
            ClassificationResult with category and confidence.
        """
        if isinstance(embedding, np.ndarray):
            tensor = torch.from_numpy(embedding).float()
        else:
            tensor = embedding.float()

        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self._device)

        self._classifier.eval()
        with torch.no_grad():
            logits = self._classifier(tensor).squeeze(0)
            probs = F.softmax(logits, dim=0)

        scores = {
            self._categories[i].value: float(probs[i].item())
            for i in range(len(self._categories))
        }

        best_idx = int(probs.argmax().item())
        return ClassificationResult(
            category=self._categories[best_idx],
            confidence=float(probs[best_idx].item()),
            all_scores=scores,
            features_used=["embedding_classifier"],
        )

    def classify(
        self,
        task: str,
        embedding: np.ndarray | torch.Tensor | None = None,
    ) -> ClassificationResult:
        """
        Classify a task using the best available method.

        Uses embedding-based classification if an embedding is provided,
        otherwise falls back to keyword rules.

        Args:
            task: Task description text.
            embedding: Optional task embedding.

        Returns:
            ClassificationResult.
        """
        if embedding is not None:
            return self.classify_embedding(embedding)
        return self.classify_text(task)
