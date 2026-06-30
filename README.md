# Voice of Customer (VoC) Insight Engine

Live demo: https://walmartvoc.streamlit.app/

The VoC Insight Engine turns raw product reviews into a ranked, actionable list
of issues for merchandising teams. It scores every review for sentiment, breaks
each one down by aspect (fit, quality, style, comfort, price, shipping), tags the
negative aspects as business issues, assigns a 1 to 5 severity, and runs an
optional LLM deep-dive on the most severe reviews to produce a root cause and a
recommended action.

It is free and open source, and runs fully offline by default. The demo uses the
[Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
dataset (22,641 reviews).

## Features

- Sentiment scoring on every review with VADER.
- Aspect-based sentiment. Each review is split into clauses and scored per
  aspect, so one review can be positive on style and negative on fit at the same
  time.
- Issue tagging that fires only when an aspect is actually negative, so a tag
  means a problem, not just a topic. "The fit is perfect" is not a sizing issue.
- Severity scoring from 1 to 5, derived from sentiment, star rating, breadth of
  complaints, and red-flag phrases.
- Optional LLM deep-dive that returns a summary, severity, aspect breakdown, and
  a recommended action for the worst reviews. It falls back to a rules engine
  when no model is available.
- A modular `voc/` package with a pytest suite of 12 tests that run without a
  network or an LLM.

## How it works

The pipeline is hybrid and cost-aware. Running an LLM on every review would be
slow and expensive, so VADER and the aspect, tagging, and severity passes score
all reviews in milliseconds for free. The LLM is reserved for a deep-dive on only
the most severe negative reviews. This is a common pattern for controlling the
cost of AI in production.

Data flow: load, VADER sentiment, aspects, tags, severity, then an optional LLM
deep-dive.

## Architecture

```
voc/
├── config.py      # aspect taxonomy and LLM backend settings
├── loading.py     # CSV load and flexible column mapping
├── sentiment.py   # VADER sentiment pass
├── aspects.py     # aspect-based sentiment: split clauses, score per aspect
├── severity.py    # 1 to 5 urgency score
├── tagging.py     # negative aspects to business issue tags
├── llm.py         # hybrid deep-dive: OpenAI-compatible client and offline fallback
├── pipeline.py    # orchestration, KPIs, CSV export
└── schemas.py     # typed result objects
app.py             # Streamlit dashboard, a thin UI over voc/
tests/             # pytest suite, runs fully offline
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python -m streamlit run app.py
```

Click Use Sample Data, or upload a CSV with a column that looks like Review Text.
Department, rating, and other columns are detected automatically when present.

## Configuration: the LLM deep-dive

The app works immediately with the offline rules engine. To enable the LLM
deep-dive, configure the backend through environment variables in a `.env` file
(copy `.env.example` to `.env`). The client is OpenAI-compatible, so the same code
works with three backends:

- Local Ollama, free with no signup. Install [Ollama](https://ollama.com) and run
  `ollama pull llama3.2`. The app auto-detects the local server.
- Groq free tier. Set the base URL, key, and model. Works on a deployed demo.
- Google Gemini free tier, through its OpenAI-compatible endpoint.

See `.env.example` for ready-made configurations.

## Customizing

Two files hold the business logic and are meant to be edited:

1. `voc/config.py`, the `ASPECTS` taxonomy. Replace the apparel keywords with
   electronics, grocery, or another category to retarget the engine.
2. `voc/severity.py`, the `score_severity` function. It sets how urgency is
   computed from sentiment, star rating, breadth of complaints, and red-flag
   phrases.

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

The suite covers column mapping, aspect splitting, issue tagging, severity
escalation, the pipeline, and the offline deep-dive. All 12 tests run without a
network or an LLM.

## License

© 2026 LEMINE MBARECK. MIT License, see [LICENSE](LICENSE).
