"""Severity scoring — how urgently should the merch team act on a review?

This is intentionally a small, opinionated function. It encodes a business
judgement (how to weigh sentiment vs. star rating vs. specific red-flag words)
and there are several defensible ways to do it, which makes it a natural place
to inject your own domain knowledge. See README "Make it your own".
"""

from __future__ import annotations

import pandas as pd

from .config import CRITICAL_PHRASES


def _isnan(x) -> bool:
    try:
        return x != x  # NaN is the only value not equal to itself
    except Exception:
        return False


def score_severity(
    sentiment_score: float,
    rating=None,
    n_negative_aspects: int = 0,
    text: str = "",
) -> int:
    """Return an integer urgency from 1 (minor) to 5 (critical).

    Default heuristic:
      * start from how negative the overall sentiment is,
      * escalate when the star rating is low,
      * escalate per *additional* negative aspect (broad dissatisfaction),
      * hard-escalate when a critical phrase (e.g. "ripped", "refund") appears.
    """
    if sentiment_score <= -0.6:
        severity = 4
    elif sentiment_score <= -0.3:
        severity = 3
    elif sentiment_score <= -0.05:
        severity = 2
    else:
        severity = 1

    if rating is not None and not _isnan(rating):
        if rating <= 1:
            severity += 2
        elif rating <= 2:
            severity += 1

    severity += max(0, n_negative_aspects - 1)

    low = str(text).lower()
    if any(phrase in low for phrase in CRITICAL_PHRASES):
        severity += 1

    return max(1, min(5, severity))


def add_severity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["severity"] = df.apply(
        lambda r: score_severity(
            r["sentiment_score"],
            r.get("rating"),
            r.get("n_negative_aspects", 0),
            r["review_text"],
        ),
        axis=1,
    )
    return df
