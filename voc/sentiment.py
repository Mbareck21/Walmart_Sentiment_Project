"""Fast, free, whole-review sentiment via NLTK VADER.

This is the cheap first pass of the hybrid pipeline: every review is scored in
microseconds so the expensive LLM deep-dive can be reserved for the negatives.
"""

from __future__ import annotations

from functools import lru_cache

import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from .config import NEG_THRESHOLD, POS_THRESHOLD


@lru_cache(maxsize=1)
def get_analyzer() -> SentimentIntensityAnalyzer:
    """Return a cached VADER analyzer, downloading the lexicon once if needed."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    return SentimentIntensityAnalyzer()


def score_sentiment(text: str) -> float:
    """VADER compound score in [-1, 1]."""
    return get_analyzer().polarity_scores(str(text))["compound"]


def label_sentiment(score: float) -> str:
    if score >= POS_THRESHOLD:
        return "Positive"
    if score <= NEG_THRESHOLD:
        return "Negative"
    return "Neutral"


def add_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add `sentiment_score` and `sentiment_label` columns (non-mutating)."""
    df = df.copy()
    df["sentiment_score"] = df["review_text"].map(score_sentiment)
    df["sentiment_label"] = df["sentiment_score"].map(label_sentiment)
    return df
