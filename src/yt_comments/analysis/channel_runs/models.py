from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ChannelRunSummary:
    channel_id: str
    started_at_utc: datetime
    finished_at_utc: datetime
    video_ids: tuple[str, ...]
    video_count: int
    comment_count: int
    error_count: int
    video_limit: int | None
    comment_limit: int | None
    published_after: datetime | None
    published_before: datetime | None