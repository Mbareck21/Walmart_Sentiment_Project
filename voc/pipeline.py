"""Pipeline orchestration: turn raw reviews into an enriched, actionable frame.

`enrich()` runs only the free, offline passes (sentiment -> aspects -> tags ->
severity) so it is fast and deterministic. The expensive LLM deep-dive is run
separately by the UI on just the top negative reviews (`top_negative`), which is
what makes the hybrid approach cheap.
"""

from __future__ import annotations

import pandas as pd

from .aspects import add_aspects
from .schemas import AspectScore
from .sentiment import add_sentiment
from .severity import add_severity
from .tagging import add_tags


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Run the free offline enrichment passes in order."""
    df = add_sentiment(df)
    df = add_aspects(df)
    df = add_tags(df)
    df = add_severity(df)
    return df


def summary_metrics(df: pd.DataFrame) -> dict:
    """Headline KPIs for the dashboard."""
    total = len(df)
    neg = df[df["sentiment_label"] == "Negative"]
    return {
        "total": total,
        "avg_sentiment": float(df["sentiment_score"].mean()) if total else 0.0,
        "negative": int(len(neg)),
        "negative_pct": (len(neg) / total * 100) if total else 0.0,
        "aspect_issues": int(df["n_negative_aspects"].sum()),
        "critical": int((df["severity"] >= 4).sum()),
    }


def top_negative(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """The n most actionable negative reviews: most severe, then most negative."""
    negs = df[df["sentiment_label"] == "Negative"].copy()
    return negs.sort_values(
        ["severity", "sentiment_score"], ascending=[False, True]
    ).head(n)


def to_export_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten list / dataclass columns to strings so the frame is CSV-safe."""
    out = df.copy()
    if "aspects" in out.columns:
        out["aspect_breakdown"] = out["aspects"].map(
            lambda items: "; ".join(
                f"{a.aspect}:{a.label}({a.score})" for a in items
            )
        )
        out = out.drop(columns=["aspects"])
    for col in ["aspects_mentioned", "negative_aspects", "issue_tags"]:
        if col in out.columns:
            out[col] = out[col].map(
                lambda v: ", ".join(v) if isinstance(v, list) else v
            )
    return out
