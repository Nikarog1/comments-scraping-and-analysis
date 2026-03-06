
# analysis/tfidf — Gold v2 TF-IDF Stage

## Overview

This stage computes corpus-level TF-IDF statistics from preprocessed Silver comments.
It transforms cleaned comment text into weighted token features suitable for:

- corpus analysis
- ranking
- downstream ML models
- feature store ingestion

This stage is deterministic and configuration-driven.

Scope note:
- Gold v2 is per-video only (no global cross-video corpus state yet).

---

## Input Contract (Silver Layer)

**Expected location**  
    <data_root>/silver/<video_id>/comments.parquet

**Required columns**

| column name  | type   | description                           |
| ------------ | ------ | ------------------------------------- |
| `text_clean` | string | Preprocessed, normalized comment text |

**Assumptions**

- `text_clean` is already cleaned (URLs replaced, normalized whitespace, Unicode normalized, etc.)
- No further heavy preprocessing (lemmatization, stemming) is performed here
- Empty or null texts are allowed and ignored
- Silver contains `preprocess_version`, which is propagated to Gold

---

## Output Contract (Gold v2)

**Expected location**  
    <data_root>/gold/tfidf/<video_id>/keywords.parquet

**Grain**
- One row per video
- The artifact represents the full comment corpus of a single video

---

## Schema

### Top-level fields (metadata + reproducibility)

| field                | type            | description                                           |
| -------------------- | --------------- | ----------------------------------------------------- |
| `video_id`           | string          | YouTube video ID                                      |
| `created_at_utc`     | timestamp (UTC) | Artifact creation time                                |
| `silver_path`        | string          | Path to Silver input                                  |
| `preprocess_version` | string          | Version of Silver preprocessing logic                 |
| `config_hash`        | string          | TF-IDF configuration fingerprint (stable hash)        |

### Corpus counters (sanity checks)

| field                 | type  | description                                                          |
| --------------------- | ----- | -------------------------------------------------------------------- |
| `row_count`           | int64 | Total rows scanned in Silver                                         |
| `empty_text_count`    | int64 | Rows ignored (null/empty or became empty after token filters)        |
| `doc_count_non_empty` | int64 | N = number of non-empty documents used for DF/IDF calculations       |
| `vocab_size`          | int64 | Tokens remaining after df filtering, before top-K cutoff             |

### Config fields (explicit, self-describing artifacts)

Tokenization / filtering:

| field             | type   | description                                   |
| ----------------- | ------ | --------------------------------------------- |
| `top_k`           | int32  | Number of keywords returned                   |
| `min_token_len`   | int32  | Minimum token length                          |
| `keep_numeric`    | bool   | Keep numeric tokens if true                   |
| `keep_stopwords`  | bool   | Keep stopwords if true                        |
| `lang`            | string | Stopwords language code (e.g. `en`)           |
| `lowercase`       | bool   | Lowercase tokens before counting              |

DF threshold configuration (raw + resolved):

| field        | type   | description                                                                 |
| ------------ | ------ | --------------------------------------------------------------------------- |
| `min_df_raw` | string | User value (e.g. `2` or `0.01`)                                             |
| `max_df_raw` | string | User value (e.g. `0.9` or `500`)                                            |
| `min_df_abs` | int64  | Resolved absolute df threshold used (fraction resolved against N)           |
| `max_df_abs` | int64  | Resolved absolute df threshold used (fraction resolved against N)           |

Algorithm modes (explicit versioning of math):

| field      | type   | description                                  |
| ---------- | ------ | -------------------------------------------- |
| `tf_mode`  | string | For v2: `norm`                               |
| `idf_mode` | string | For v2: `smooth_log_plus1_ln`                |

### Keywords (nested list)

| field      | type                  | description                                         |
| ---------- | --------------------- | --------------------------------------------------- |
| `keywords` | list<struct>          | Ranked list of top-K keyword records                |

Each `keywords` element is a struct:

| field    | type    | description                                  |
| -------- | ------- | -------------------------------------------- |
| `token`  | string  | Token text                                   |
| `score`  | float64 | Final TF-IDF score                           |
| `avg_tf` | float64 | Average normalized TF across documents       |
| `idf`    | float64 | Inverse document frequency                   |
| `df`     | int64   | Document frequency                            |

---

## Mathematical Definition

**Document**
- One document = one comment (after filtering)

**N**
- N = number of non-empty documents (comments) after tokenization + filters

**Document Frequency**
- df(t) = |{ d ∈ D : t ∈ d }|

**TF (normalized)**
- tf(t, d) = count(t, d) / |d|

**Average TF**
- avg_tf(t) = (1/N) * Σ_d tf(t, d)

**IDF (sklearn-style smoothed, natural log)**
- idf(t) = ln((1 + N) / (1 + df(t))) + 1

**Final score**
- score(t) = avg_tf(t) * idf(t)

---

## Ranking + Tie-breaking (Determinism)

Keywords are sorted by:

1. `score` descending
2. `df` descending
3. `token` ascending (lexicographic)

This ensures deterministic outputs even when float scores tie.

---

## Configuration

TF-IDF behavior is fully determined by `TfidfConfig`.
All configuration parameters are hashed via `config_hash` using stable serialization.

---

## Determinism Guarantees

This stage guarantees:

- Same input + same config → identical output
- No randomness
- UTC timestamps
- Stable ranking + tie-breaking as specified above
- No external runtime downloads (no dynamic stopword fetching)

---

## Performance Model

- Reads Silver Parquet in streaming batches
- Reads only required columns
- Memory usage is dominated by vocabulary size:
  - O(V) for `df[token]` and `sum_tf_norm[token]`
- Suitable for per-video corpora; no global corpus maintained in v2

---

## Design Decisions

- No scikit-learn black box in v2 (explicit formulas + deterministic implementation)
- Stopwords are explicitly versioned and controlled via config
- IDF smoothing chosen for stability and comparability with common tooling
- Arrow/Parquet used for schema enforcement and efficient scans
- One-pass streaming scan (no second pass required for v2)

---

## Future Extensions

Potential evolutions:

- BM25 weighting
- Per-channel corpus aggregation
- Global multi-video corpus (Gold v3)
- Language-specific tokenization
- Vector normalization (L2)
- Incremental corpus df store + updates

---

## Versioning Strategy

- `preprocess_version`
  - changes when Silver text preprocessing changes (normalization, URL replacement, etc.)
- `config_hash`
  - changes when TF-IDF parameters change (thresholds, filters, top_k, etc.)
- `tf_mode` / `idf_mode`
  - changes when mathematical definition changes

Schema changes require a major version bump for this Gold stage.


