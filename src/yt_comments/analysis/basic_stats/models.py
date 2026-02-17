from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence



@dataclass(frozen=True, slots=True)
class BasicStatsConfig:
    """
    Deterministic configuration for gold v1 basic stats
    """
    top_n_tokens: int = 30
    min_token_len: int = 2
    drop_numeric_tokens: bool = True
    lowercase: bool = True # added for explicity and reproducibility
    
@dataclass(frozen=True, slots=True)
class TopToken:
    token: str
    count: int
    
@dataclass(frozen=True, slots=True)
class BasicStats:
    video_id: str
    silver_path: str 
    created_at_utc: datetime
    preprocess_version: str
    config_hash: str
    row_count: int
    empty_text_count: int
    total_token_count: int
    unique_token_count: int
    top_tokens: Sequence[TopToken]
    
    