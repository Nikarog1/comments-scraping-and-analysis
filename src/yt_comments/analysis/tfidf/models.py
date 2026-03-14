from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class TfidfConfig:
    top_k: int = 30
    min_token_len: int = 2
    drop_numeric_tokens: bool = True
    lowercase: bool = True
    drop_stopwords: bool = True
    stopwords_lang: str = "en"
    stopwords_hash: str = "dummy"
    normalization: str = "none" # stemming
    min_df: int | float = 2 # token must appear in at least 2 comments
    max_df: int | float = 0.9 # drop tokens appearing in >90% comments 
    tf_mode: str = "norm"
    idf_mode: str = "smooth_log_plus1_ln"
    ngram_range: tuple[int, int] = (1, 1)
    min_ngram_df: int = 2 # ensures to extract only repeating ngrams
    
@dataclass(frozen=True, slots=True)    
class TfidfKeyword:
    token: str
    score: float
    idf: float
    avg_tf: float
    df: int
    
@dataclass(frozen=True, slots=True)
class TfidfKeywords:
    video_id: str
    created_at_utc: datetime
    silver_path: str
    preprocess_version: str
    artifact_version: str
    config_hash: str
    row_count: int
    empty_text_count: int
    doc_count_non_empty: int
    vocab_size: int
    min_df_raw: str
    max_df_raw: str
    min_df_abs: int
    max_df_abs: int
    config: TfidfConfig # decided to add config info too for better dubugging and visibility 
    keywords: tuple[TfidfKeyword, ...] # corrected from Sequential; to have it immutable


    
    