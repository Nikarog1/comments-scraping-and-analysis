from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from yt_comments.analysis.tfidf.models import TfidfConfig



@dataclass(frozen=True, slots=True)
class DistinctiveKeyword:
    """
    Keyword distinctive to a video compared to the channel.

    lift = video_score / channel_score
    """
    token: str
    video_score: float
    channel_score: float
    lift: float
    video_df: int
    channel_df: int
    
@dataclass(frozen=True, slots=True)
class DistinctiveKeywords:
    channel_id: str
    video_id: str
    created_at_utc: datetime
    preprocess_version: str
    artifact_version: str
    config_hash: str
    config: TfidfConfig
    keyword_count: int
    keywords: tuple[DistinctiveKeyword, ...] 