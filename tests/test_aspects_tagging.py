from voc.aspects import analyze_aspects
from voc.severity import score_severity
from voc.tagging import issue_tags, primary_issue


def _by_aspect(aspects):
    return {a.aspect: a for a in aspects}


def test_mixed_review_splits_sentiment_by_aspect():
    """The headline capability: one review, opposite polarity per aspect."""
    a = _by_aspect(
        analyze_aspects("I love the fabric but the fit is terrible and way too tight.")
    )
    assert a["Quality & Material"].label == "Positive"
    assert a["Fit & Sizing"].label == "Negative"


def test_positive_fit_is_not_flagged_as_issue():
    """Regression vs. the original app, where the keyword 'fit' alone tagged a
    Sizing Issue even in praise."""
    aspects = analyze_aspects("The fit is perfect and true to size.")
    assert issue_tags(aspects) == []
    assert primary_issue(aspects) == "General"


def test_negative_fit_tagged_sizing():
    aspects = analyze_aspects("The fit is awful, runs way too small.")
    assert "Sizing Issue" in issue_tags(aspects)


def test_severity_escalates_with_low_rating_and_critical_phrase():
    base = score_severity(-0.4, rating=3, n_negative_aspects=1)
    assert score_severity(-0.4, rating=1, n_negative_aspects=1) > base
    crit = score_severity(-0.4, rating=3, n_negative_aspects=1, text="it ripped on day one")
    assert base < crit <= 5
