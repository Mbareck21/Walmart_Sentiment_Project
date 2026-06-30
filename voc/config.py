"""Central configuration: the aspect taxonomy and the (free) LLM backend settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


# The aspect taxonomy: each aspect maps to the keywords that signal it in a review.
# This is the most domain-specific part of the system; rewrite it to adapt to a
# different product category.
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

# Maps a negative aspect to the business issue tag a merchandising team acts on.
ASPECT_TO_ISSUE: dict[str, str] = {
    "Fit & Sizing": "Sizing Issue",
    "Quality & Material": "Quality Issue",
    "Style & Appearance": "Style/Design",
    "Comfort": "Comfort Issue",
    "Price & Value": "Pricing",
    "Shipping & Delivery": "Supply Chain",
}

# Phrases that escalate severity regardless of overall sentiment.
CRITICAL_PHRASES: list[str] = [
    "ripped", "tore", "torn", "fell apart", "broke", "broken", "refund",
    "never again", "waste of money", "returned", "defective", "hole",
]

POS_THRESHOLD = 0.05
NEG_THRESHOLD = -0.05


@dataclass
class LLMConfig:
    """OpenAI-compatible backend settings; defaults to a local Ollama server."""

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
        return f"{self.model} @ {urlparse(self.base_url).netloc}"


def get_llm_config() -> LLMConfig:
    return LLMConfig()
