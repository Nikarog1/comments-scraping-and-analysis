from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from yt_comments.analysis.basic_stats.models import BasicStatsConfig, TopToken
    


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
    top_tokens: tuple[TopToken, ...]