"""Aspect-based sentiment — the granular core of the engine.

Whole-review sentiment hides mixed feedback: "love the fabric but the fit is
terrible" nets out near neutral, which is useless to a merchandiser. This module
splits each review into clauses, decides which aspect(s) each clause is about,
and scores those clauses independently — so we can report *what* customers
dislike, not merely *that* they are unhappy.
"""

from __future__ import annotations

import re

import pandas as pd

from .config import ASPECTS
from .schemas import AspectScore
from .sentiment import get_analyzer, label_sentiment

# Split on sentence terminators AND contrastive conjunctions, because sentiment
# usually flips at "but / however / although" inside a single sentence.
_CLAUSE_RE = re.compile(
    r"(?<=[.!?])\s+|\s+(?:but|however|although|though|yet)\s+", re.IGNORECASE
)


def _compile(keywords: list[str]) -> list[re.Pattern]:
    # Word-boundary match so "fit" does not fire on "outfit".
    return [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in keywords]


_ASPECT_PATTERNS: dict[str, list[re.Pattern]] = {
    aspect: _compile(kws) for aspect, kws in ASPECTS.items()
}


def split_clauses(text: str) -> list[str]:
    parts = _CLAUSE_RE.split(str(text))
    return [p.strip() for p in parts if p and p.strip()]


def analyze_aspects(text: str) -> list[AspectScore]:
    """Return one AspectScore per aspect actually mentioned in the review."""
    analyzer = get_analyzer()
    # Score every clause once, then reuse for each aspect.
    clause_scores = [
        (clause, analyzer.polarity_scores(clause)["compound"])
        for clause in split_clauses(text)
    ]

    results: list[AspectScore] = []
    for aspect, patterns in _ASPECT_PATTERNS.items():
        hits = [
            (clause, score)
            for clause, score in clause_scores
            if any(pat.search(clause) for pat in patterns)
        ]
        if not hits:
            continue
        mean = sum(score for _, score in hits) / len(hits)
        # Evidence = the clause that best illustrates the aspect's polarity.
        evidence = (min if mean < 0 else max)(hits, key=lambda cs: cs[1])[0]
        results.append(
            AspectScore(
                aspect=aspect,
                score=round(mean, 4),
                label=label_sentiment(mean),
                evidence=evidence,
            )
        )
    return results


def add_aspects(df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-review aspect analysis plus convenience scalar columns."""
    df = df.copy()
    df["aspects"] = df["review_text"].map(analyze_aspects)
    df["aspects_mentioned"] = df["aspects"].map(lambda a: [x.aspect for x in a])
    df["negative_aspects"] = df["aspects"].map(
        lambda a: [x.aspect for x in a if x.label == "Negative"]
    )
    df["n_negative_aspects"] = df["negative_aspects"].map(len)
    return df


def aspect_long_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Explode per-review aspect lists into a tidy (review, aspect) frame.

    Drives the aspect bar chart and the department x aspect heatmap.
    """
    rows = []
    for idx, row in df.iterrows():
        for a in row["aspects"]:
            rows.append(
                {
                    "review_index": idx,
                    "department_name": row.get("department_name", "Unknown"),
                    "aspect": a.aspect,
                    "score": a.score,
                    "label": a.label,
                }
            )
    return pd.DataFrame(
        rows, columns=["review_index", "department_name", "aspect", "score", "label"]
    )
