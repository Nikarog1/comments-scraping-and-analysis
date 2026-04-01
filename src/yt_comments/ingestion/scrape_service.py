from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from yt_comments.ingestion.models import Comment
from yt_comments.ingestion.youtube_api_client import YouTubeApiClient
from yt_comments.storage.bronze_comments_repository import JSONLCommentsRepository



@dataclass(slots=True)
class ScrapeResult:
    video_id: str
    saved_count: int
    path: Path
    
@dataclass(slots=True)
class ScrapeCommentsService:
    client: YouTubeApiClient
    repo: JSONLCommentsRepository

    def run(self, video_id: str, *, overwrite: bool = True, limit: int | None = None) -> ScrapeResult:
        comments: list[Comment] = []
        for c in self.client.fetch_comments(video_id):
            comments.append(c)
            if limit is not None and len(comments) >= limit:
                break
        path = self.repo.save(video_id, comments, overwrite=overwrite)
        return ScrapeResult(video_id=video_id, saved_count=len(comments), path=path)