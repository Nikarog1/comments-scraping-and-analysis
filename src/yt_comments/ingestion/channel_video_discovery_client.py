from __future__ import annotations

from typing import Iterable, Protocol

from yt_comments.ingestion.models import ChannelVideo, ChannelVideoDiscovery


class ChannelVideoDiscoveryClient(Protocol):
    def discover_videos(self, request: ChannelVideoDiscovery) -> Iterable[ChannelVideo]:
        ...