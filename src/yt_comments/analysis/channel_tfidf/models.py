from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from yt_comments.analysis.tfidf.models import TfidfConfig, TfidfKeyword



@dataclass(frozen=True, slots=True)
class ChannelTfidfKeywords:
    channel_id: str
    video_ids: tuple[str, ...]
    created_at_utc: datetime
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