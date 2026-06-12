"""VoC Insight Engine — granular Voice-of-Customer analytics.

Pipeline: load -> VADER sentiment -> aspect-based sentiment -> issue tagging ->
severity -> optional hybrid LLM deep-dive (free, local Ollama by default).
"""

__version__ = "2.0.0"
