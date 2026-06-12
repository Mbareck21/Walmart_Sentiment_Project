import pandas as pd
import pytest

from voc.loading import load_reviews


def test_flexible_column_mapping():
    df = pd.DataFrame(
        {"Review Text": ["great"], "Department Name": ["Tops"], "Rating": [5]}
    )
    out = load_reviews(df)
    assert {"review_text", "department_name", "rating"} <= set(out.columns)
    assert out.loc[0, "department_name"] == "Tops"
    assert out.loc[0, "rating"] == 5


def test_missing_review_column_raises():
    with pytest.raises(ValueError):
        load_reviews(pd.DataFrame({"foo": ["bar"]}))


def test_department_fallback_and_blank_rows_dropped():
    out = load_reviews(pd.DataFrame({"review": ["nice", "", "   "]}))
    assert (out["department_name"] == "Unknown").all()
    assert len(out) == 1  # blank / whitespace-only reviews removed
