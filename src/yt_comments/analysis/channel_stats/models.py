from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ChannelTokenStat:
    token: str
    count: int
    

@dataclass(frozen=True, slots=True)
class ChannelTokenStatsConfig:
    """
    Deterministic configuration for gold v4 channel basic stats
    """
    top_n_tokens: int = 30
    min_token_len: int = 2
    drop_numeric_tokens: bool = True
    lowercase: bool = True 
    drop_stopwords: bool = True
    stopwords_lang: str = "en"
    stopwords_hash: str = "dummy"
    normalization: str = "none" # stemming
    

@dataclass(frozen=True, slots=True)
class ChannelTokenStats:
    channel_id: str
    video_ids: tuple[str, ...]
    created_at_utc: datetime
    preprocess_version: str
    config_hash: str
    row_count: int
    empty_text_count: int
    total_token_count: int
    unique_token_count: int
    top_tokens: tuple[ChannelTokenStat, ...]