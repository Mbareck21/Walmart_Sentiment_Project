"""CSV loading and column normalisation, kept UI-free so it stays testable."""

from __future__ import annotations

import pandas as pd

# Engine column -> substrings (case-insensitive) that identify it; first match wins.
_COLUMN_HINTS: dict[str, list[str]] = {
    "review_text": ["review text", "review", "comment", "feedback"],
    "department_name": ["department"],
    "class_name": ["class"],
    "division_name": ["division"],
    "rating": ["rating", "stars"],
    "title": ["title"],
    "age": ["age"],
    "recommended": ["recommended"],
}


def _match_columns(df: pd.DataFrame) -> dict[str, str]:
    rename: dict[str, str] = {}
    used: set = set()
    for target, hints in _COLUMN_HINTS.items():
        if target in df.columns:
            continue
        for hint in hints:
            match = None
            for col in df.columns:
                if col in used:
                    continue
                name = str(col).strip().lower()
                if name == hint or hint in name:
                    match = col
                    break
            if match is not None:
                rename[match] = target
                used.add(match)
                break
    return rename


def load_reviews(source) -> pd.DataFrame:
    """Read a CSV (path/file-like) or DataFrame and normalise the schema.

    Raises ValueError if no review-text column is found.
    """
    df = source if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    df = df.rename(columns=_match_columns(df))

    if "review_text" not in df.columns:
        raise ValueError(
            "Could not find a review-text column. Expected a column whose name "
            "contains 'Review Text' (or 'review', 'comment', 'feedback')."
        )

    if "department_name" not in df.columns:
        df["department_name"] = "Unknown"
    df["department_name"] = df["department_name"].fillna("Unknown").astype(str)

    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"].str.len() > 0]
    df = df[~df["review_text"].str.lower().isin({"nan", "none"})]
    return df.reset_index(drop=True)
