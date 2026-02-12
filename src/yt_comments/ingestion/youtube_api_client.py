from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from yt_comments.ingestion.youtube_client import YouTubeClient
from yt_comments.ingestion.models import Comment



@dataclass(slots=True)
class YouTubeApiClient(YouTubeClient):
    api_key: str
    
    def fetch_comments(self, video_id: str):
        raise NotImplementedError("Not implemented. Use StubYouTubeClient for now")