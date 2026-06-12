"""Central configuration for the VoC Insight Engine.

Everything tunable lives here: the aspect taxonomy that powers granular
aspect-based sentiment, and the (free) LLM backend settings used for the
hybrid deep-dive. All defaults are free / offline-friendly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # optional: load a local .env if present
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


# ---------------------------------------------------------------------------
# Aspect taxonomy  (DOMAIN KNOWLEDGE — this is the part worth customising)
# ---------------------------------------------------------------------------
# Each aspect maps to the keywords/phrases that signal it in a review. The
# aspect engine (voc/aspects.py) splits each review into clauses and scores the
# sentiment of ONLY the clauses that mention an aspect — so a single review can
# be positive on "Style" and negative on "Fit & Sizing" at the same time.
#
# This taxonomy is the most domain-specific part of the system. To adapt the
# engine to, say, electronics instead of apparel, this is the dict you rewrite.
ASPECTS: dict[str, list[str]] = {
    "Fit & Sizing": [
        "fit", "fits", "fitted", "size", "sizing", "tight", "loose", "snug",
        "baggy", "small", "large", "big", "tiny", "oversized", "true to size",
        "runs small", "runs large", "length", "short", "long", "petite",
        "narrow", "wide",
    ],
    "Quality & Material": [
        "quality", "fabric", "material", "cheap", "flimsy", "thin", "sturdy",
        "durable", "tear", "torn", "rip", "ripped", "seam", "stitching",
        "stitch", "fell apart", "pilling", "pill", "wrinkle", "wrinkled",
        "see-through", "see through", "sheer", "fray", "frayed", "shrink",
        "shrunk",
    ],
    "Style & Appearance": [
        "color", "colour", "style", "design", "pattern", "print", "look",
        "looks", "cute", "ugly", "flattering", "unflattering", "beautiful",
        "gorgeous", "pretty", "elegant", "shape", "cut",
    ],
    "Comfort": [
        "comfortable", "comfy", "uncomfortable", "soft", "scratchy", "itchy",
        "stiff", "breathable", "heavy", "lightweight", "cozy",
    ],
    "Price & Value": [
        "price", "priced", "expensive", "overpriced", "worth", "value", "deal",
        "cost", "pricey", "affordable", "bargain", "money",
    ],
    "Shipping & Delivery": [
        "shipping", "shipped", "delivery", "delivered", "arrived", "package",
        "packaging", "late", "delayed", "delay", "shipment", "return",
        "returned", "refund", "exchange", "tracking",
    ],
}

# Maps a NEGATIVE aspect to the business "issue tag" merchandising teams act on.
ASPECT_TO_ISSUE: dict[str, str] = {
    "Fit & Sizing": "Sizing Issue",
    "Quality & Material": "Quality Issue",
    "Style & Appearance": "Style/Design",
    "Comfort": "Comfort Issue",
    "Price & Value": "Pricing",
    "Shipping & Delivery": "Supply Chain",
}

# High-risk phrases that escalate severity regardless of overall score.
CRITICAL_PHRASES: list[str] = [
    "ripped", "tore", "torn", "fell apart", "broke", "broken", "refund",
    "never again", "waste of money", "returned", "defective", "hole",
]

# Sentiment thresholds (VADER compound score).
POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05


@dataclass
class LLMConfig:
    """Settings for the hybrid deep-dive backend (OpenAI-compatible).

    Defaults point at a local, FREE Ollama server. The same code works with the
    Groq or Google Gemini free tiers by overriding base_url / model / api_key
    via environment variables (see .env.example).
    """

    base_url: str = field(
        default_factory=lambda: os.getenv("VOC_LLM_BASE_URL", "http://localhost:11434/v1")
    )
    api_key: str = field(default_factory=lambda: os.getenv("VOC_LLM_API_KEY", "ollama"))
    model: str = field(default_factory=lambda: os.getenv("VOC_LLM_MODEL", "llama3.2"))
    timeout: float = field(default_factory=lambda: float(os.getenv("VOC_LLM_TIMEOUT", "30")))
    max_deep_dive: int = field(
        default_factory=lambda: int(os.getenv("VOC_LLM_MAX_DEEP_DIVE", "25"))
    )

    @property
    def host_label(self) -> str:
        host = (
            self.base_url.replace("/v1", "")
            .replace("http://", "")
            .replace("https://", "")
        )
        return f"{self.model} @ {host}"


def get_llm_config() -> LLMConfig:
    return LLMConfig()
