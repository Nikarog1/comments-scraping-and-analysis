# YouTube Comments Scraping & Analysis

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

A Python 3.11 project for **scraping YouTube comments and performing deterministic NLP analysis** without LLMs.

The system is built as a **data pipeline with Bronze / Silver / Gold layers**, focusing on:

- reproducibility
- deterministic outputs
- clean architecture
- scalable data processing
- minimal external dependencies

---

# Installation

Editable install on Windows:

```bash
py -m pip install -e .
py -m pip install -e ".[dev]"
```

---

# Project Architecture

The pipeline follows a **Bronze → Silver → Gold** design.

YouTube API -> Bronze (raw comments) -> Silver (cleaned comments) -> Gold (analytical artifacts)

Each stage has a clearly defined responsibility:

| Layer | Responsibility |
|------|---------------|
| Bronze | Raw data ingestion from the YouTube API |
| Silver | Deterministic preprocessing and normalization |
| Gold | Analytical artifacts derived from the corpus |

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
yt-comments stats <video_id>
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

**CLI**
```bash
yt_comments tfidf <video_id>
```


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

---

# Project Structure

src/
    yt_comments/
        analysis/
            basic_stats/
            tfidf/
        ingestion/
        preprocessing/
        storage/
        cli/
tests/


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

---

# Future Extensions

Potentional next stages:

- cross-video TF-IDF corpus (Gold v3)
- stopwords enhancments & comment-domain stopwords 
- add keyword blacklist
- additional languages support
- topic clustering
- bigrams
- semantic embeddings 
- channel-level analytics 
- incremental corpus statistics 

---

# License 

MIT License









