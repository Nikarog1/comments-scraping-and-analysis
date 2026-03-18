from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from yt_comments.ingestion.channel_video_discovery_client import ChannelVideoDiscoveryClient
from yt_comments.ingestion.models import ChannelVideo, ChannelVideoDiscovery


@dataclass(slots=True)
class ChannelVideoDiscoveryResult:
    video_count: int
    videos: Iterable[ChannelVideo]
    

@dataclass(slots=True)
class ChannelVideoDiscoveryService:
    client: ChannelVideoDiscoveryClient
    request: ChannelVideoDiscovery
    
    def run(self) -> ChannelVideoDiscoveryResult:
        videos: list[ChannelVideo] = []
        for video in self.client.discover_videos(request=self.request):           
            videos.append(video)      
        video_count = len(videos)
        return ChannelVideoDiscoveryResult(video_count=video_count, videos=videos)