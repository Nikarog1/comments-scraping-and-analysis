from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Protocol

from yt_comments.ingestion.models import ChannelVideo, ChannelVideoDiscovery


class ChannelVideoDiscoveryClient(Protocol):
    def discover_videos(self, request: ChannelVideoDiscovery) -> Iterable[ChannelVideo]:
        ...

@dataclass(slots=True)    
class StubChannelVideoDiscoveryClient:
    """
    Temporary client for testing and / or if API key isn't provided
    """
    
    def discover_videos(self, request: ChannelVideoDiscovery) -> Iterable[ChannelVideo]:
        return [
            ChannelVideo(
                video_id="v1",
                channel_id=request.channel_id,
                title="Example video 1",
                published_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
            ),
            ChannelVideo(
                video_id="v2",
                channel_id=request.channel_id,
                title="Example video 2",
                published_at=datetime(2026, 1, 2, 10, 45, tzinfo=timezone.utc)
            ),
            ChannelVideo(
                video_id="v3",
                channel_id=request.channel_id,
                title="Example video 3",
                published_at=datetime(2026, 2, 1, 15, 5, tzinfo=timezone.utc)
            ),
        ]
    