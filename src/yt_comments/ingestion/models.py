from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime



@dataclass(frozen=True, slots=True)
class Comment:
    video_id: str
    comment_id: str
    text: str
    author: str | None = None
    like_count: int | None = None
    published_at: datetime | None = None
    is_reply: bool = False

@dataclass(frozen=True, slots=True)
class ChannelVideo:
    video_id: str
    channel_id: str
    title: str
    published_at: datetime | None = None
    view_count: str | None = None
    
@dataclass(frozen=True, slots=True)
class ChannelVideoDiscovery:
    channel_id: str
    published_after: datetime | None = None
    published_before: datetime | None = None
    limit: int | None = None
    