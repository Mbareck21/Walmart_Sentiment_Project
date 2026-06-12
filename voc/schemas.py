"""Typed result objects shared across the engine and the UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class AspectScore:
    """Sentiment for one aspect within one review."""

    aspect: str
    score: float           # VADER compound over the aspect's clauses, -1..1
    label: str             # Positive / Neutral / Negative
    evidence: str = ""     # representative clause that mentioned the aspect


@dataclass
class DeepDive:
    """Structured root-cause analysis for a single (usually negative) review.

    Produced either by the LLM backend or by the offline rules fallback; the
    `source` field records which, so the UI can be transparent about it.
    """

    summary: str
    root_cause: str
    recommended_action: str
    severity: int                              # 1 (minor) .. 5 (critical)
    aspects: list[AspectScore] = field(default_factory=list)
    source: str = "offline"                    # "offline" or "llm:<model>"

    def to_dict(self) -> dict:
        return asdict(self)
