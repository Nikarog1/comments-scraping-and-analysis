# YouTube Comments Scraping & Analysis

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

Deterministic pipeline for scraping and analyzing YouTube comments using classical NLP (no LLMs nor external libs as NLTK).

The project is designed to be reproducible, inspectable, and easy to reason about.  
All intermediate data is materialized to disk.

NOTE: the project at this stage supports only English comments!

---

## Demo

If you want to quickly see results without running the full pipeline, check the notebooks in:

```
notebooks/
```

---

## Pipeline

```
YouTube API
   ↓
Bronze  → raw comments (JSONL)
   ↓
Silver  → cleaned text (Parquet)
   ↓
Gold    → analytical artifacts (stats, TF-IDF, channel analytics)
```

- Each step writes results to disk  
- Later steps never modify earlier ones  
- No hidden state  

---

## Quick Start

<video_id> can be either a YouTube video ID or a full video URL.

<channel_id> can be a full URL, a handle, or an API ID. Use the same format consistently throughout the pipeline, except for the API ID, which can be used at any stage.

**Single video analysis:**
```bash
# Scrape comments
yt_comments scrape <video_id>

# Preprocess
yt_comments preprocess <video_id>

# Build analytics
yt_comments stats <video_id>
yt_comments tfidf <video_id>

# OPTIONAL: cross-video corpus
yt_comments corpus
yt_comments tfidf <video_id> --use-corpus
```

**Channel videos analysis:**
```bash
# Discover videos
yt_comments discover-videos <channel_id>

# Scrape comments
yt_comments scrape-channel <channel_id>

# Preprocess
yt_comments preprocess-channel <channel_id>

# Build analytics
yt_comments channel-stats <channel_id>
yt_comments channel-tfidf <channel_id>

# Final report (no recomputation)
yt_comments report-channel <channel_id>
```

---

## Data layout

```
data/
  bronze/
    <video_id>.jsonl

  silver/
    <video_id>/comments.parquet

  gold/
    basic_stats/
    tfidf/
    corpus/
    channel_runs/
    channel_stats/
    channel_tfidf/
    distinctive_keywords/
```

---

## Reproducibility

All artifacts store:

- preprocess_version — version of text preprocessing  
- config_hash — hash of analysis parameters  

This allows exact reproduction of results and comparison across runs.

---

## Layers

**Bronze**
Raw comments stored as JSONL exactly as returned by the API.

**Silver**
Cleaned and normalized text:
- lowercasing
- URL replacement
- repeated character normalization
- token filtering

Stored as Parquet for efficient processing.

**Gold**
Analytical artifacts:
- basic statistics
- TF-IDF keywords
- channel-level aggregations
- distinctive keywords per video

---

## TF-IDF

TF-IDF is computed deterministically using the following definitions.

**Document Frequency**

df(t) = number of documents containing token t

**Normalized Term Frequency**

tf(t, d) = count(t, d) / document_length

**Average TF**

avg_tf(t) = (1 / N) * Σ tf(t, d)

**Smoothed IDF (scikit-learn style)**

idf(t) = ln((1 + N) / (1 + df(t))) + 1

**Final score**

score(t) = avg_tf(t) * idf(t)

Where:

N = number of non-empty documents

Supports:
- document-level TF-IDF
- optional global corpus for IDF
- n-grams (unigrams, bigrams)

---

## Design decisions

- deterministic pipeline (no randomness, no LLMs)
- append-only channel runs
- CLI is orchestration only (no business logic)
- artifacts are immutable once written

---

## Notes

- Focus is on classical NLP (tokenization, TF-IDF, statistics)
- Designed for clarity and maintainability over feature complexity
- Support of English comments only

---

# License 

MIT License










