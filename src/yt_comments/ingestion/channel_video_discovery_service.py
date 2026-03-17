from __future__ import annotations

from typing import Iterable

from yt_comments.ingestion.channel_video_discovery_client import ChannelVideoDiscoveryClient
from yt_comments.ingestion.models import ChannelVideo, ChannelVideoDiscovery


class ChannelVideoDiscoveryService:
    def __init__(self, client: ChannelVideoDiscoveryClient) -> None:
        self._client = client
        
    def discover_videos(self, request: ChannelVideoDiscovery) -> Iterable[ChannelVideo]:
        return self._client.discover_videos(request)