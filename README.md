# Voice of Customer (VoC) Insight Engine

**[🚀 View Live Demo](https://walmartvoc.streamlit.app/)**

Turn thousands of raw product reviews into a ranked, *actionable* punch list for
merchandising teams — using a **hybrid, cost-aware AI pipeline** that is free and
open-source end to end.

> Merch teams drown in reviews and miss the quality, sizing, and supply-chain
> problems buried in the noise. This engine reads the haystack and hands back the
> needles: *what* customers complain about, *where* it concentrates, *how urgent*
> it is, and *what to do next*.

The demo uses the **[Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)** dataset (22,641 reviews).

---

## What makes this version different

This is a ground-up rebuild of a single-file prototype into a tested, modular
engine. The headline upgrades:

| Capability | Original | This version |
|---|---|---|
| Sentiment | One VADER score per review | VADER **+ aspect-based** sentiment per clause |
| Granularity | Whole-review polarity | Per-aspect scores: *Fit, Quality, Style, Comfort, Price, Shipping* |
| Issue tagging | Keyword presence (so "perfect **fit**" → *Sizing Issue* 🐛) | Tagged only when the aspect is actually **negative** |
| Root cause | — | **Hybrid LLM deep-dive**: summary, severity (1–5), recommended action |
| Urgency | — | **Severity score** from sentiment + rating + red-flag phrases |
| Architecture | One 200-line `app.py` | `voc/` package + **12 passing tests** |
| Cost | Free | **Still free** — local Ollama by default, offline rules fallback |

**Why "hybrid"?** Running an LLM on all 22k reviews is slow and expensive.
Instead, VADER scores *everything* in milliseconds for free, and the LLM is
reserved for a deep-dive on only the worst negative reviews — the classic
production pattern for controlling AI cost.

### Tags mean problems, not topics
The original tagged any review containing `fit`/`size` as a *Sizing Issue* — even
*"the fit is perfect."* Here, a clause is scored for sentiment **before** it
becomes an issue tag, so tags mean *problems*, not *topics*. There's a regression
test for exactly this (`tests/test_aspects_tagging.py`).

---

## Architecture

```
voc/
├── config.py      # aspect taxonomy + LLM backend settings (the tunable knobs)
├── loading.py     # CSV load + flexible column mapping (raises, doesn't print)
├── sentiment.py   # fast VADER pass (cheap first stage)
├── aspects.py     # ★ aspect-based sentiment: split clauses, score per aspect
├── severity.py    # 1–5 urgency heuristic (your domain logic lives here)
├── tagging.py     # negative aspects → business issue tags
├── llm.py         # hybrid deep-dive: OpenAI-compatible client + offline fallback
├── pipeline.py    # orchestration + KPIs + CSV export
└── schemas.py     # typed result objects
app.py             # Streamlit dashboard (thin UI over voc/)
tests/             # pytest suite (runs fully offline)
```

The data flow: **load → VADER → aspects → tags → severity → (optional) LLM deep-dive.**

---

## Setup

```bash
# 1. create & activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt

# 3. run
python -m streamlit run app.py
```

Then click **🚀 Use Sample Data**, or upload any CSV with a column that looks
like *Review Text* (department, rating, etc. are auto-detected if present).

### Optional: enable the AI deep-dive (free, local)
The app works immediately with a free offline rules engine. To enable the **LLM**
deep-dive at $0 with no signup, install [Ollama](https://ollama.com) and pull a
small model:

```bash
ollama pull llama3.2          # ~2 GB, free, runs locally
```

That's it — the app auto-detects the local Ollama server. The client is
**OpenAI-compatible**, so you can instead point it at the **Groq** or **Google
Gemini** free tiers (which also work on a deployed demo) by copying
`.env.example` to `.env` and setting the backend. See that file for ready-made
configs.

---

## Make it your own

Two files encode *business judgement* and are designed to be customised — these
are the most valuable places to add your own domain knowledge:

1. **`voc/config.py` → `ASPECTS`** — the aspect taxonomy. Swap the apparel
   keywords for electronics, grocery, etc. to retarget the whole engine.
2. **`voc/severity.py` → `score_severity()`** — how urgency is computed from
   sentiment, star rating, breadth of complaints, and red-flag phrases. There are
   many defensible weightings; the default is a starting point.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

The suite covers column mapping, aspect splitting, the issue-tagging regression,
severity escalation, the pipeline, and the offline deep-dive — all without a
network or an LLM.

## License
© 2026 LEMINE MBARECK. MIT License — see [LICENSE](LICENSE).
