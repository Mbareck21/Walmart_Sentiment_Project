"""VoC Insight Engine — Streamlit dashboard.

Thin UI layer over the `voc` package. All analytics live in `voc/`; this file
only loads data, caches the heavy work, and draws the dashboard.
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import streamlit as st

from voc import llm as llm_mod
from voc.aspects import aspect_long_frame
from voc.config import get_llm_config
from voc.loading import load_reviews
from voc.pipeline import enrich, summary_metrics, to_export_df, top_negative

st.set_page_config(page_title="VoC Insight Engine", layout="wide", page_icon="🛒")

SAMPLE_PATH = "data/Reviews.csv"
SEVERITY_LABELS = {1: "Minor", 2: "Low", 3: "Medium", 4: "High", 5: "Critical"}


# --------------------------------------------------------------------------- #
# Cached heavy work
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, persist="disk")
def process(raw_df: pd.DataFrame):
    """Run the offline enrichment once per dataset and derive chart frames.

    persist="disk" keeps the result across restarts, so only the very first
    analysis of a given dataset pays the full cost.
    """
    enriched = enrich(raw_df)
    return enriched, aspect_long_frame(enriched), summary_metrics(enriched)


@st.cache_data(show_spinner=False)
def load_sample(path: str) -> pd.DataFrame:
    """Cache the sample CSV read so reruns (e.g. picking a review) stay fast."""
    return load_reviews(path)


@st.cache_data(show_spinner=False)
def cached_deep_dive(text, score, rating, prefer_llm, model, base_url):
    """Per-review deep-dive, cached so re-selecting a review is instant.

    model/base_url are part of the cache key so switching backend invalidates.
    """
    cfg = get_llm_config()
    cfg.model, cfg.base_url = model, base_url
    return llm_mod.deep_dive(
        text, score, rating, prefer_llm=prefer_llm, cfg=cfg
    ).to_dict()


@st.cache_data(show_spinner="Checking LLM backend…")
def backend_status(base_url, model):
    cfg = get_llm_config()
    cfg.base_url, cfg.model = base_url, model
    return llm_mod.backend_available(cfg)


def severity_badge(sev: int) -> str:
    color = {1: "🟢", 2: "🟢", 3: "🟡", 4: "🟠", 5: "🔴"}.get(sev, "⚪")
    return f"{color} {SEVERITY_LABELS.get(sev, sev)} ({sev}/5)"


# --------------------------------------------------------------------------- #
# Sidebar — backend controls
# --------------------------------------------------------------------------- #
cfg = get_llm_config()
with st.sidebar:
    st.header("⚙️ Analysis Engine")
    st.caption("Fast VADER pass on every review, optional LLM deep-dive on the worst.")
    available = backend_status(cfg.base_url, cfg.model)
    if available:
        st.success(f"LLM backend online · `{cfg.host_label}`")
    else:
        st.warning(
            f"LLM backend offline (`{cfg.host_label}`).\n\n"
            "Deep-dives use the **free rules engine**. "
            "Add a backend key in `.env` to enable the LLM — see `.env.example`."
        )
    use_llm = st.toggle("Use LLM for deep-dive", value=available, disabled=not available)
    max_dd = st.slider("Max reviews to deep-dive", 5, 50, cfg.max_deep_dive, step=5)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🛒 Voice of Customer (VoC) Insight Engine")

with st.expander("ℹ️ How this tool works", expanded=False):
    st.markdown(
        """
**The problem.** Merchandising teams drown in thousands of reviews and miss the
quality, sizing, and supply-chain issues buried in the noise.

**The approach — a hybrid, cost-aware pipeline:**
1. **Fast sentiment (VADER):** every review is scored in milliseconds, for free.
2. **Aspect-based sentiment:** each review is split into clauses and scored *per
   aspect* (Fit, Quality, Style, Comfort, Price, Shipping) — so we capture
   *"love the fabric **but** the fit is off"* instead of one blurry average.
3. **Issue tagging & severity:** negative aspects become actionable tags and a
   1–5 urgency score (sentiment + rating + red-flag phrases).
4. **AI deep-dive (optional):** an LLM analyses only the worst reviews for root
   cause and a recommended action — falling back to a rules engine when no model
   is available, so it always works.
        """
    )

# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
col_load1, col_load2 = st.columns([2, 1])
with col_load1:
    uploaded = st.file_uploader("Upload a review CSV", type=["csv"])
with col_load2:
    st.write(
        "[Sample: Women's E-Commerce Clothing Reviews]"
        "(https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)"
    )
    if st.button("🚀 Use Sample Data", width="stretch"):
        st.session_state["use_sample"] = True

raw_df = None
try:
    if uploaded is not None:
        raw_df = load_reviews(uploaded)
        st.session_state.pop("use_sample", None)  # an upload overrides the sample
    elif st.session_state.get("use_sample"):
        if os.path.exists(SAMPLE_PATH):
            raw_df = load_sample(SAMPLE_PATH)
        else:
            st.error(f"Sample data not found at '{SAMPLE_PATH}'.")
except ValueError as exc:
    st.error(str(exc))

if raw_df is None:
    st.info("⬆️ Upload a CSV or click **Use Sample Data** to begin.")
    st.stop()

with st.spinner("Running analysis (sentiment → aspects → tags → severity)…"):
    df, long_df, kpis = process(raw_df)

st.success(f"Analysis complete — {kpis['total']:,} reviews processed.")

# --------------------------------------------------------------------------- #
# KPI row
# --------------------------------------------------------------------------- #
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Feedback", f"{kpis['total']:,}")
k2.metric("Avg Sentiment", f"{kpis['avg_sentiment']:.2f}")
k3.metric("Negative Reviews", f"{kpis['negative']:,}", f"{kpis['negative_pct']:.1f}%",
          delta_color="inverse")
k4.metric("Aspect Issues Flagged", f"{kpis['aspect_issues']:,}")
k5.metric("Critical (sev ≥ 4)", f"{kpis['critical']:,}", delta_color="inverse")
st.divider()

# --------------------------------------------------------------------------- #
# Aspect-based sentiment (the granular headline)
# --------------------------------------------------------------------------- #
st.subheader("🎯 Aspect-Based Sentiment")
st.caption("What customers actually praise and complain about — not just an overall score.")

ac1, ac2 = st.columns(2)
with ac1:
    agg = (
        long_df.groupby("aspect")["score"].mean().reset_index().sort_values("score")
    )
    fig_aspect = px.bar(
        agg, x="score", y="aspect", orientation="h", color="score",
        color_continuous_scale="RdYlGn", range_color=[-1, 1],
        title="Average sentiment by aspect (red = unhappy)",
    )
    fig_aspect.update_layout(yaxis_title="", xaxis_title="avg VADER score")
    st.plotly_chart(fig_aspect, width="stretch")

with ac2:
    neg_long = long_df[long_df["label"] == "Negative"]
    if not neg_long.empty:
        pivot = neg_long.pivot_table(
            index="department_name", columns="aspect", values="review_index",
            aggfunc="count", fill_value=0,
        )
        fig_heat = px.imshow(
            pivot, color_continuous_scale="Reds", aspect="auto", text_auto=True,
            title="Negative mentions: department × aspect",
        )
        fig_heat.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_heat, width="stretch")
    else:
        st.info("No negative aspect mentions detected.")

# --------------------------------------------------------------------------- #
# Department risk + root cause (corrected vs. the original keyword tagging)
# --------------------------------------------------------------------------- #
st.subheader("⚠️ Risk & Root Cause")
rc1, rc2 = st.columns(2)
neg_reviews = df[df["sentiment_label"] == "Negative"]
with rc1:
    dept_risk = (
        neg_reviews.groupby("department_name").size().reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    fig_bar = px.bar(
        dept_risk, x="department_name", y="count", color="count",
        color_continuous_scale="Reds", title="Negative feedback volume by department",
    )
    fig_bar.update_layout(xaxis_title="", yaxis_title="negative reviews")
    st.plotly_chart(fig_bar, width="stretch")
with rc2:
    if not neg_reviews.empty:
        tag_counts = neg_reviews["issue_tag"].value_counts().reset_index()
        tag_counts.columns = ["issue_tag", "count"]
        fig_pie = px.pie(
            tag_counts, names="issue_tag", values="count", hole=0.4,
            title="Primary driver of negative sentiment",
        )
        st.plotly_chart(fig_pie, width="stretch")
    else:
        st.info("No negative reviews to break down.")

# --------------------------------------------------------------------------- #
# AI deep-dive (on-demand, per selected review)
# --------------------------------------------------------------------------- #
st.subheader("🤖 AI Root-Cause Deep-Dive")
top = top_negative(df, max_dd)
if top.empty:
    st.info("No negative reviews to analyse.")
else:
    src = f"LLM · {cfg.host_label}" if (use_llm and available) else "Free offline rules engine"
    st.caption(f"Analysing the {len(top)} most severe negative reviews · source: **{src}**")

    options = {
        f"#{idx} · {row['department_name']} · sev {row['severity']} · "
        f"{row['review_text'][:70]}…": idx
        for idx, row in top.iterrows()
    }
    choice = st.selectbox("Pick a flagged review to analyse", list(options.keys()))
    sel_idx = options[choice]
    sel = df.loc[sel_idx]
    rating = None if "rating" not in df.columns or pd.isna(sel.get("rating")) else float(sel["rating"])

    with st.spinner("Generating structured analysis…"):
        dd = cached_deep_dive(
            sel["review_text"], float(sel["sentiment_score"]), rating,
            use_llm and available, cfg.model, cfg.base_url,
        )

    st.markdown(f"> {sel['review_text']}")
    d1, d2 = st.columns([1, 2])
    with d1:
        st.metric("Severity", severity_badge(dd["severity"]))
        st.caption(f"source: `{dd['source']}`")
        for a in dd["aspects"]:
            mark = {"Negative": "🔴", "Positive": "🟢"}.get(a["label"], "⚪")
            st.write(f"{mark} **{a['aspect']}** · {a['score']:+.2f}")
    with d2:
        st.markdown(f"**Summary.** {dd['summary']}")
        st.markdown(f"**Root cause.** {dd['root_cause']}")
        st.success(f"**Recommended action.** {dd['recommended_action']}")

# --------------------------------------------------------------------------- #
# Urgent action list + export
# --------------------------------------------------------------------------- #
st.subheader("📋 Urgent Action List")
st.caption("Negative reviews ranked by severity. Sortable; export the full enriched report below.")
action_cols = ["department_name", "severity", "issue_tag", "sentiment_score", "review_text"]
action = (
    neg_reviews[action_cols]
    .sort_values("severity", ascending=False)
    .rename(columns={"sentiment_score": "sentiment", "department_name": "department"})
)
st.dataframe(action, width="stretch", height=380)

st.download_button(
    "📥 Download enriched report (CSV)",
    data=to_export_df(df).to_csv(index=False).encode("utf-8"),
    file_name="voc_enriched_report.csv",
    mime="text/csv",
)

st.divider()
st.caption("© 2026 LEMINE MBARECK · VoC Insight Engine — hybrid VADER + LLM, free & open-source.")
