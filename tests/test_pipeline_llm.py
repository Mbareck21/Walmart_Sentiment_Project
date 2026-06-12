import pandas as pd

from voc.llm import offline_deep_dive
from voc.pipeline import enrich, summary_metrics, to_export_df, top_negative

SAMPLE = pd.DataFrame(
    {
        "review_text": [
            "Absolutely love it, gorgeous and so comfy.",
            "Terrible quality — the fabric is awful and it ripped on the first wear.",
            "Fits great but the color was a bit dull.",
        ],
        "department_name": ["Tops", "Dresses", "Tops"],
        "rating": [5, 1, 4],
    }
)


def test_enrich_adds_expected_columns():
    df = enrich(SAMPLE)
    expected = {
        "sentiment_score", "sentiment_label", "aspects", "issue_tag",
        "issue_tags", "severity", "n_negative_aspects",
    }
    assert expected <= set(df.columns)
    assert len(df) == 3


def test_summary_metrics_keys():
    m = summary_metrics(enrich(SAMPLE))
    assert {"total", "avg_sentiment", "negative", "aspect_issues", "critical"} <= set(m)
    assert m["total"] == 3


def test_top_negative_ranks_worst_first():
    top = top_negative(enrich(SAMPLE), 5)
    assert not top.empty
    assert "ripped" in top.iloc[0]["review_text"].lower()


def test_offline_deep_dive_structure():
    dd = offline_deep_dive(
        "The fabric ripped and I want a refund.", sentiment_score=-0.6, rating=1
    )
    assert dd.source == "offline"
    assert 1 <= dd.severity <= 5
    assert dd.recommended_action  # non-empty playbook action
    assert isinstance(dd.aspects, list)


def test_export_is_csv_safe():
    out = to_export_df(enrich(SAMPLE))
    assert "aspects" not in out.columns  # dataclass column flattened
    assert "aspect_breakdown" in out.to_csv(index=False)
