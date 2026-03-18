from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional 



@dataclass(frozen=True, slots=True)
class Comment:
    video_id: str
    comment_id: str
    text: str
    author: Optional[str] = None
    like_count: Optional[int] = None
    published_at: Optional[datetime] = None
    is_reply: bool = False

@dataclass(frozen=True, slots=True)
class ChannelVideo:
    video_id: str
    channel_id: str
    title: str
    published_at: Optional[datetime] = None
    view_count: Optional[str] = None
    
@dataclass(frozen=True, slots=True)
class ChannelVideoDiscovery:
    channel_id: str
    published_after: Optional[datetime] = None
    published_before: Optional[datetime] = None
    limit: Optional[int] = None
    