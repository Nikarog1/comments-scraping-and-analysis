# YouTube Comments Scraping & Analysis

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

A Python 3.11 deterministic pipeline for **scraping and analyzing YouTube comments using classical NLP (no LLMs).**

The system designed for reproducibility and inspectable intermediate data.

NOTE: the project at this stage supports only English comments!

---

# Installation

Editable install on Windows:

```bash
py -m pip install -e .
py -m pip install -e ".[dev]"
```

---

# Quick Start

Use --help to get more information on each command and its arguments.

<video_id> can be either pure YouTube video id or just a normal URL.

<channel_id> can be pure URL, a handle, or API id. Note that you should use the same format in the whole pipeline except API id which can be used in ANY point.

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
py -m yt_comments discover-videos <channel_id>

# Scrape comments
py -m yt_comments scrape-channel <channel_id>

# Preprocess
py -m yt_comments preprocess-channel <channel_id>

# Build analytics
py -m yt_comments channel-stats <channel_id>
py -m yt_comments channel-tfidf <channel_id>

# Final report (no recomputation)
py -m yt_comments report-channel <channel_id>
```

---

# Pipeline

YouTube API
   ↓
Bronze  → raw comments (JSONL)
   ↓
Silver  → cleaned text (Parquet)
   ↓
Gold    → analytical artifacts (stats, TF-IDF, channel analytics)

- Each layer is materialized to disk
- No hidden state — everything is reproducible
- Later steps never mutate earlier data

---

# Project Architecture

The pipeline follows a **Bronze → Silver → Gold** design.

YouTube API -> Bronze (raw comments) -> Silver (cleaned comments) -> Gold (analytical artifacts)

## Data Layout

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

## Data Layers

### Bronze — Raw Ingestion

The Bronze layer stores **raw YouTube API responses** with minimal transformation.

**Location**

data/bronze/<video_id>/comments.jsonl

**CLI**
```bash
yt_comments scrape <video_id>
```

**Properties**

- append-only storage
- fail-fast corruption detection
- preserves original API payload

**Purpose**

- reliable ingestion
- debugging and replay capability
- source-of-truth storage


### Silver — Cleaned Comments

The Silver layer converts raw comments into **analysis-ready text**.

**Output**

data/silver/<video_id>/comments.parquet

**CLI**
```bash
yt_comments preprocess <video_id>
```

**Transformations**

- Unicode normalization (NFKC)
- URL replacement
- whitespace normalization
- optional lowercasing

**Schema**

| column | description |
|------|-------------|
| text_clean | normalized comment text |

**Design principles**

- deterministic preprocessing
- streaming Parquet writing
- one file per video


## Gold Layer

The Gold layer contains **derived analytical artifacts** computed from Silver data.

Characteristics:

- one artifact per video
- deterministic results
- fully reproducible
- configuration versioned

**Each artifact is stored in its own directory:**

data/gold/<analysis>/<video_id>

Current Gold stages:

### Gold v1 — Basic Statistics

Computes corpus statistics such as:

- total comments
- empty comments
- token counts
- vocabulary size
- most frequent tokens

**Output**

data/gold/basic_stats/<video_id>/stats.parquet

**CLI**
```bash
yt_comments stats <video_id>
```

### Gold v2 — TF-IDF Keywords

Extracts **top keywords from a video's comment corpus** using TF-IDF.

**Document unit**

1 comment = 1 document

**Corpus**

All comments for a single video 

**Output**

data/gold/tfidf/<video_id>/keywords.parquet

Each artifact includes:

- corpus metadata
- configuration hash
- document statistics
- ranked TF-IDF keywords

Configuration hashes include:

- analysis configuration
- stopword vocabulary
- keyword quality rules

This ensures that artifacts produced with different configurations cannot be accidentally mixed.

**CLI**
```bash
yt_comments tfidf <video_id>
```

### N-gram Support

TF-IDF supports both:

- unigrams
- bigrams (any n-grams)

Bigrams are generated deterministically during feature extraction.

Example keywords:

- black cat
- rain sounds
- keyboard tapping

### Gold v3 — Cross-Video Corpus

Gold v3 introduces a **global corpus artifact** used for cross-video TF-IDF.

Instead of computing document frequency only within a single video, the pipeline can compute document frequency across multiple videos.

This improves keyword extraction by identifying terms that are unique to a specific video compared to the broader corpus.

**Document unit**

1 video = 1 document

**Meaning**

df(token) = number of videos containing the token

**Output**

data/gold/corpus_df/corpus.parquet

**CLI**

```bash
yt_comments corpus
```

The corpus artifact stores:

- token
- document frequency
- total video count
- configuration metadata

TF-IDF can optionally use this artifact to compute global IDF values.


## TF-IDF Definition

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


## Determinism Guarantees

The pipeline guarantees:

- identical outputs for identical inputs and configuration
- no randomness
- stable token ordering
- explicit configuration hashing
- UTC timestamps

All Gold artifacts are **self-describing and reproducible**.

### Global TF-IDF (v3)

If a global corpus artifact is available, TF-IDF can compute IDF using cross-video statistics.

Local video:

- TF computed from comments of the target video

Global corpus:

- IDF computed using document frequencies across videos

Formula remains:

score(t) = avg_tf(t) * idf(t)

but:

N = number of videos in the corpus

---

# Project Structure

## Project structure

```
comments-scraping-and-analysis/
├── src/
│   └── yt_comments/
│       ├── analysis/
│       │   ├── basic_stats/
│       │   ├── tfidf/
│       │   ├── corpus/
│       │   ├── features.py
│       │   └── keyword_quality.py
│       ├── cli/
│       ├── ingestion/
│       ├── nlp/
│       ├── preprocessing/
│       └── storage/
└── tests/
```


**Design rules**

- **analysis** - domain logic
- **ingestion** - scrape service
- **nlp** - stopwords (TO BE ENHANCED)
- **preprocessing** - preprocessing
- **storage** - infrastructure
- **cli** - composition root


## Design Principles

The project prioritizes:

- reproducibility-first design
- deterministic pipelines
- minimal dependencies
- explicit configuration
- clear separation of concerns
- streaming processing where justified


## Feature Generation

Tokenization and feature generation are centralized in:

analysis/features.py

Responsibilities:

- tokenization
- stopword filtering
- n-gram generation
- configuration hashing
- preprocessing metadata validation

This module is shared across:

- basic_stats
- tfidf
- corpus

Centralizing feature generation ensures that all analysis stages operate on the **same feature space**, preventing drift between artifacts.


## Stopwords

Stopwords are defined in:

nlp/stopwords.py

Three vocabularies are maintained:

- linguistic stopwords
- YouTube-specific conversational stopwords
- sentiment vocabulary

Sentiment vocabulary is **not removed during preprocessing**, because it may be useful for future sentiment analysis.

Stopwords are included in the TF-IDF configuration hash to ensure reproducibility when the vocabulary changes.


## Keyword Quality Filtering

After TF-IDF scoring, keywords are filtered using deterministic rules to improve readability.

Examples of filtered tokens:

- sentiment unigrams (e.g. "amazing", "great")
- praise bigrams (e.g. "great video", "amazing video")

This filtering step removes conversational praise phrases while preserving topical keywords.

Filtering rules are versioned via:

KEYWORD_QUALITY_VERSION

---

# License 

MIT License









