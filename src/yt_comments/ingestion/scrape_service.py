from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from yt_comments.ingestion.models import Comment
from yt_comments.ingestion.youtube_client import YouTubeClient
from yt_comments.storage.comments_repository import JSONLCommentsRepository



@dataclass(slots=True)
class ScrapeResult:
    video_id: str
    saved_count: int
    path: Path
    
@dataclass(slots=True)
class ScrapeCommentsService:
    client: YouTubeClient
    repo: JSONLCommentsRepository

    def run(self, video_id: str, *, overwrite: bool = True) -> ScrapeResult:
        comments = list(self.client.fetch_comments(video_id))
        path = self.repo.save(video_id, comments, overwrite=overwrite)
        return ScrapeResult(video_id=video_id, saved_count=len(comments), path=path)