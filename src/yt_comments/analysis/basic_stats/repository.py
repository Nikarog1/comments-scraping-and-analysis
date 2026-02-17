from __future__ import annotations

from typing import Protocol

from yt_comments.analysis.basic_stats.models import BasicStats



class BasicStatsRepository(Protocol):
    """
    Protocol for storage of gold v1 basic stats
    """
    def save(self, stats: BasicStats) -> None:
        ...
        
    def load(self, video_id: str) -> BasicStats:
        ...