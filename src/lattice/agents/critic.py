"""
Critic agent for evaluating outputs using Constitutional AI principles.

Implements evaluation inspired by Constitutional AI (Bai et al., 2022):
the critic reviews agent outputs against a set of principles (a
"constitution") and provides structured feedback with scores. This
enables iterative refinement loops where outputs are critiqued and
improved until they meet quality standards.

The critic can operate in two modes:
    1. **Score mode**: Assigns numeric scores on configurable dimensions.
    2. **Revision mode**: Suggests specific revisions to improve the output.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from lattice.agents.base import AgentCapabilities, AgentResult, AgentStatus, BaseAgent


class ConstitutionalPrinciple(BaseModel):
    """A single principle in the critic's constitution."""

    name: str
    description: str
    weight: float = Field(default=1.0, ge=0.0, le=10.0)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class CritiqueScore(BaseModel):
    """Score for a single evaluation dimension."""

    principle: str
    score: float = Field(ge=0.0, le=1.0)
    explanation: str
    passes_threshold: bool


class CritiqueResult(BaseModel):
    """Complete evaluation result from the critic."""

    scores: list[CritiqueScore] = Field(default_factory=list)
    overall_score: float = Field(ge=0.0, le=1.0)
    passes: bool
    revision_suggestions: list[str] = Field(default_factory=list)
    summary: str = ""


DEFAULT_CONSTITUTION: list[ConstitutionalPrinciple] = [
    ConstitutionalPrinciple(
        name="helpfulness",
        description="The output directly addresses the user's request and provides useful information",
        weight=2.0,
        threshold=0.7,
    ),
    ConstitutionalPrinciple(
        name="accuracy",
        description="The output contains factually correct information without hallucinations",
        weight=2.0,
        threshold=0.8,
    ),
    ConstitutionalPrinciple(
        name="safety",
        description="The output does not contain harmful, dangerous, or unethical content",
        weight=3.0,
        threshold=0.9,
    ),
    ConstitutionalPrinciple(
        name="coherence",
        description="The output is well-structured, logically consistent, and easy to understand",
        weight=1.0,
        threshold=0.6,
    ),
    ConstitutionalPrinciple(
        name="completeness",
        description="The output fully addresses all aspects of the request",
        weight=1.5,
        threshold=0.7,
    ),
]


class CriticAgent(BaseAgent):
    """
    Constitutional AI critic for evaluating agent outputs.

    Reviews outputs against a set of principles and produces structured
    scores and revision suggestions. Can be used in a refinement loop
    where another agent revises its output based on the critique.

    Args:
        agent_id: Unique identifier.
        name: Human-readable name.
        constitution: Principles to evaluate against.
        llm_provider: LLM provider for generating evaluations.
        model: Model to use for evaluation.
        passing_score: Minimum overall score to pass.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        name: str = "CriticAgent",
        constitution: list[ConstitutionalPrinciple] | None = None,
        llm_provider: Any = None,
        model: str = "gpt-4o",
        passing_score: float = 0.7,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            name=name,
            description="Evaluates outputs against constitutional principles",
            capabilities=AgentCapabilities(
                supported_task_types=["evaluation", "critique", "review"],
            ),
        )
        self._constitution = constitution or DEFAULT_CONSTITUTION
        self._llm = llm_provider
        self._model = model
        self._passing_score = passing_score

    @property
    def constitution(self) -> list[ConstitutionalPrinciple]:
        return list(self._constitution)

    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Evaluate the given text/output against the constitution.

        The task should be the text to evaluate. Context can contain
        the original request for reference.

        Args:
            task: The text to evaluate.
            context: Optional context (e.g., {"original_request": "..."}).

        Returns:
            AgentResult with CritiqueResult in the output.
        """
        start = time.perf_counter()
        self._status = AgentStatus.BUSY

        try:
            critique = await self._evaluate(task, context)
            latency = (time.perf_counter() - start) * 1000

            result = AgentResult(
                agent_id=self._agent_id,
                task=task[:200],
                output=critique.model_dump(),
                success=True,
                latency_ms=latency,
                metadata={
                    "overall_score": critique.overall_score,
                    "passes": critique.passes,
                    "num_principles": len(self._constitution),
                },
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            result = AgentResult(
                agent_id=self._agent_id,
                task=task[:200],
                output=None,
                success=False,
                error=str(e),
                latency_ms=latency,
            )
        finally:
            self._status = AgentStatus.IDLE
            self._record_execution(result)

        return result

    async def _evaluate(
        self,
        text: str,
        context: dict[str, Any] | None,
    ) -> CritiqueResult:
        """
        Evaluate text against all constitutional principles.

        Uses LLM-based evaluation when available, otherwise falls back
        to simple heuristic scoring.
        """
        if self._llm:
            return await self._llm_evaluate(text, context)
        return self._heuristic_evaluate(text, context)

    async def _llm_evaluate(
        self,
        text: str,
        context: dict[str, Any] | None,
    ) -> CritiqueResult:
        """Evaluate using an LLM."""
        principles_desc = "\n".join(
            f"- {p.name}: {p.description} (threshold: {p.threshold})"
            for p in self._constitution
        )

        prompt = (
            "Evaluate the following output against these principles. "
            "For each principle, provide a score from 0.0 to 1.0 and a brief explanation.\n\n"
            f"Principles:\n{principles_desc}\n\n"
            f"Output to evaluate:\n{text}\n"
        )

        if context and "original_request" in context:
            prompt += f"\nOriginal request: {context['original_request']}\n"

        prompt += (
            "\nRespond in JSON: {\"scores\": [{\"principle\": \"name\", "
            "\"score\": 0.0-1.0, \"explanation\": \"...\"}], "
            "\"revision_suggestions\": [\"...\"]}"
        )

        response = await self._llm.complete(
            prompt, model=self._model, temperature=0.1
        )

        import json
        try:
            data = json.loads(response.content)
            scores = []
            for s in data.get("scores", []):
                principle = next(
                    (p for p in self._constitution if p.name == s["principle"]),
                    None,
                )
                threshold = principle.threshold if principle else 0.7
                scores.append(CritiqueScore(
                    principle=s["principle"],
                    score=s["score"],
                    explanation=s.get("explanation", ""),
                    passes_threshold=s["score"] >= threshold,
                ))

            overall = self._compute_overall_score(scores)
            return CritiqueResult(
                scores=scores,
                overall_score=overall,
                passes=overall >= self._passing_score,
                revision_suggestions=data.get("revision_suggestions", []),
            )
        except (json.JSONDecodeError, KeyError):
            return self._heuristic_evaluate(text, context)

    def _heuristic_evaluate(
        self,
        text: str,
        context: dict[str, Any] | None,
    ) -> CritiqueResult:
        """Simple heuristic evaluation when no LLM is available."""
        scores: list[CritiqueScore] = []
        suggestions: list[str] = []

        for principle in self._constitution:
            score = self._heuristic_score(text, principle)
            scores.append(CritiqueScore(
                principle=principle.name,
                score=score,
                explanation=f"Heuristic score for {principle.name}",
                passes_threshold=score >= principle.threshold,
            ))
            if score < principle.threshold:
                suggestions.append(
                    f"Improve {principle.name}: {principle.description}"
                )

        overall = self._compute_overall_score(scores)
        return CritiqueResult(
            scores=scores,
            overall_score=overall,
            passes=overall >= self._passing_score,
            revision_suggestions=suggestions,
            summary=f"Heuristic evaluation: {overall:.2f} overall",
        )

    @staticmethod
    def _heuristic_score(text: str, principle: ConstitutionalPrinciple) -> float:
        """Compute a simple heuristic score for a principle."""
        text_len = len(text)

        if principle.name == "completeness":
            # Longer responses tend to be more complete
            return min(text_len / 500.0, 1.0)
        elif principle.name == "coherence":
            # Check for sentence structure
            sentences = text.count(".") + text.count("!") + text.count("?")
            return min(sentences / 5.0, 1.0) if text_len > 0 else 0.0
        elif principle.name == "safety":
            # Very basic: check for absence of known unsafe patterns
            unsafe_patterns = ["ignore previous", "system prompt", "jailbreak"]
            has_unsafe = any(p in text.lower() for p in unsafe_patterns)
            return 0.2 if has_unsafe else 0.95
        elif principle.name == "helpfulness":
            return min(text_len / 200.0, 1.0) if text_len > 10 else 0.1
        elif principle.name == "accuracy":
            # Cannot truly assess accuracy heuristically
            return 0.7
        else:
            return 0.5

    def _compute_overall_score(self, scores: list[CritiqueScore]) -> float:
        """Weighted average of principle scores."""
        if not scores:
            return 0.0

        principle_map = {p.name: p for p in self._constitution}
        weighted_sum = 0.0
        total_weight = 0.0

        for s in scores:
            weight = principle_map.get(s.principle, ConstitutionalPrinciple(
                name=s.principle, description=""
            )).weight
            weighted_sum += s.score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0
