"""Hybrid LLM deep-dive (free, local Ollama by default).

For the relatively few negative reviews, ask an LLM to produce a structured
root-cause analysis. If no LLM backend is reachable, fall back to a
deterministic rules-based analysis so the feature still works fully offline.

The client is OpenAI-compatible, so the same code drives Ollama (default), Groq
or Google Gemini free tiers — only the env vars change (see .env.example).
"""

from __future__ import annotations

import json
import re
from typing import Optional

from .aspects import analyze_aspects
from .config import LLMConfig, get_llm_config
from .schemas import AspectScore, DeepDive
from .severity import score_severity
from .tagging import primary_issue

_SYSTEM = (
    "You are a retail Voice-of-Customer analyst. Given one product review, "
    "return STRICT JSON with keys: summary (string, <=20 words), root_cause "
    "(string), recommended_action (string: one concrete next step for a "
    "merchandising team), severity (integer 1-5), aspects (array of objects "
    "each with keys: aspect, label one of Positive/Neutral/Negative, score "
    "float between -1 and 1). Return ONLY the JSON object, no prose."
)

# Default playbook actions used by the offline fallback, keyed by issue tag.
_ACTION_BY_ISSUE = {
    "Sizing Issue": "Review size chart and fit specs with the supplier; consider "
    "re-grading or adding fit guidance to the product page.",
    "Quality Issue": "Open a quality audit with the vendor; inspect a sample batch "
    "for the defect described.",
    "Style/Design": "Share with the design/buying team; assess whether this is "
    "isolated or a pattern across the line.",
    "Comfort Issue": "Flag fabric/construction to product development; evaluate "
    "alternative materials.",
    "Pricing": "Benchmark price vs. perceived value and competitors; review promo "
    "or markdown strategy.",
    "Supply Chain": "Escalate to fulfilment/logistics; check carrier performance "
    "and packaging for this SKU.",
    "General": "Route to the category manager for manual review.",
}


def _make_client(cfg: LLMConfig):
    from openai import OpenAI

    return OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout)


def backend_available(cfg: Optional[LLMConfig] = None) -> bool:
    """Cheaply check whether an OpenAI-compatible backend is reachable."""
    cfg = cfg or get_llm_config()
    try:
        client = _make_client(LLMConfig(**{**cfg.__dict__, "timeout": min(cfg.timeout, 3)}))
        client.models.list()
        return True
    except Exception:
        return False


def _safe_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content).rstrip("`").strip()
    try:
        return json.loads(content)
    except Exception:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _parse(data: dict, review_text: str) -> DeepDive:
    aspects: list[AspectScore] = []
    for item in data.get("aspects", []) or []:
        try:
            aspects.append(
                AspectScore(
                    aspect=str(item.get("aspect", "")).strip() or "General",
                    score=float(item.get("score", 0) or 0),
                    label=str(item.get("label", "Neutral")).title(),
                )
            )
        except Exception:
            continue
    severity = max(1, min(5, int(data.get("severity", 3) or 3)))
    return DeepDive(
        summary=(str(data.get("summary", "")).strip() or review_text[:120])[:300],
        root_cause=str(data.get("root_cause", "")).strip(),
        recommended_action=str(data.get("recommended_action", "")).strip(),
        severity=severity,
        aspects=aspects,
    )


def _create(client, cfg: LLMConfig, messages, use_json: bool):
    kwargs = dict(model=cfg.model, messages=messages, temperature=0)
    if use_json:
        kwargs["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**kwargs)


def llm_deep_dive(text: str, cfg: Optional[LLMConfig] = None) -> DeepDive:
    """Call the LLM backend for a structured analysis. Raises on failure."""
    cfg = cfg or get_llm_config()
    client = _make_client(cfg)
    hint = ", ".join(a.aspect for a in analyze_aspects(text)) or "none detected"
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": f'Review: """{text}"""\nLikely aspects: {hint}\nReturn the JSON now.',
        },
    ]
    try:
        resp = _create(client, cfg, messages, use_json=True)
    except Exception:
        resp = _create(client, cfg, messages, use_json=False)  # older servers
    content = resp.choices[0].message.content or "{}"
    result = _parse(_safe_json(content), text)
    result.source = f"llm:{cfg.model}"
    return result


def offline_deep_dive(
    text: str, sentiment_score: Optional[float] = None, rating=None
) -> DeepDive:
    """Deterministic, dependency-free fallback built from the rules engine."""
    aspects = analyze_aspects(text)
    negs = [a for a in aspects if a.label == "Negative"]
    issue = primary_issue(aspects)
    score = (
        sentiment_score
        if sentiment_score is not None
        else min((a.score for a in aspects), default=0.0)
    )
    severity = score_severity(score, rating, len(negs), text)

    if negs:
        worst = min(negs, key=lambda a: a.score)
        summary = f"Customer dissatisfied with {worst.aspect.lower()}."
        root_cause = (
            f'Negative sentiment concentrated on {worst.aspect} '
            f'("{worst.evidence[:140]}").'
        )
    else:
        summary = "General dissatisfaction without a single dominant aspect."
        root_cause = "No specific negative aspect detected; overall tone is the signal."

    return DeepDive(
        summary=summary,
        root_cause=root_cause,
        recommended_action=_ACTION_BY_ISSUE.get(issue, _ACTION_BY_ISSUE["General"]),
        severity=severity,
        aspects=aspects,
        source="offline",
    )


def deep_dive(
    text: str,
    sentiment_score: Optional[float] = None,
    rating=None,
    prefer_llm: bool = True,
    cfg: Optional[LLMConfig] = None,
) -> DeepDive:
    """Try the LLM backend when requested; otherwise (or on error) use rules."""
    if prefer_llm:
        try:
            return llm_deep_dive(text, cfg)
        except Exception:
            pass
    return offline_deep_dive(text, sentiment_score, rating)
