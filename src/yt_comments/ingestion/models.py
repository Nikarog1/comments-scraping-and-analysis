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