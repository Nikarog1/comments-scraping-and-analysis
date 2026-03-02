from __future__ import annotations

from typing import Protocol

from yt_comments.analysis.tfidf.models import TfidfKeywords


class TfidfKeywordsRepository(Protocol):
    """
    Protocol for storage of gold v2 TF-IDF
    """
    def save(self, keywords: TfidfKeywords) -> None:
        ...
        
    def load(self, video_id: str) -> TfidfKeywords:
        ...