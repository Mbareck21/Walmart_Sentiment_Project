"""Issue tagging derived from aspect sentiment.

The original app tagged by raw keyword presence, so "the fit is perfect" was
flagged as a Sizing Issue. Here a tag is emitted only when the relevant aspect
is actually NEGATIVE — so tags mean problems, not topics. Reviews can carry
multiple tags ("issue_tags"); "issue_tag" keeps a single dominant label for
back-compatibility with the original pie chart.
"""

from __future__ import annotations

import pandas as pd

from .config import ASPECT_TO_ISSUE
from .schemas import AspectScore


def issue_tags(aspects: list[AspectScore]) -> list[str]:
    tags: list[str] = []
    for a in aspects:
        if a.label == "Negative" and a.aspect in ASPECT_TO_ISSUE:
            tag = ASPECT_TO_ISSUE[a.aspect]
            if tag not in tags:
                tags.append(tag)
    return tags


def primary_issue(aspects: list[AspectScore]) -> str:
    """The single most-negative issue tag, or 'General' if none qualify."""
    negs = [a for a in aspects if a.label == "Negative" and a.aspect in ASPECT_TO_ISSUE]
    if not negs:
        return "General"
    worst = min(negs, key=lambda a: a.score)
    return ASPECT_TO_ISSUE[worst.aspect]


def add_tags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["issue_tags"] = df["aspects"].map(issue_tags)
    df["issue_tag"] = df["aspects"].map(primary_issue)
    return df
