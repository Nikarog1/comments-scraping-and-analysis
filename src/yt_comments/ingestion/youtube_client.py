from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional, Protocol

from yt_comments.ingestion.models import Comment



class YouTubeClient(Protocol):
    def fetch_comments(self, video_id: str) -> Iterable[Comment]:
        """
        Return comments for a video
        """
        ...
        
@dataclass(slots=True)
class StubYouTubeClient:
    """
    Temporary client used to prove wiring (CLI -> ingestion -> storage).
    Replace with real YouTube Data API client later.
    """

    fixed_now: Optional[datetime] = None

    def fetch_comments(self, video_id: str) -> Iterable[Comment]:
        now = self.fixed_now or datetime.now(timezone.utc)

        return [
            Comment(
                video_id=video_id,
                comment_id="c1",
                text="This is a dummy top-level comment.",
                author="dummy_user_1",
                like_count=3,
                published_at=now,
                is_reply=False,
            ),
            Comment(
                video_id=video_id,
                comment_id="c2",
                text="Another dummy top-level comment.",
                author="dummy_user_2",
                like_count=0,
                published_at=now,
                is_reply=False,
            ),
            Comment(
                video_id=video_id,
                comment_id="r1",
                text="This is a dummy reply to a comment.",
                author="dummy_user_3",
                like_count=None,
                published_at=now,
                is_reply=True,
            ),
        ]