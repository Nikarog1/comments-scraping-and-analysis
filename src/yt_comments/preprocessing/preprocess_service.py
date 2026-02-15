from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import pyarrow as pa

from yt_comments.ingestion.models import Comment
from yt_comments.preprocessing.text_preprocessor import TextPreprocessor
from yt_comments.storage.protocols import BronzeCommentsReader
from yt_comments.storage.silver_comments_repository import ParquetSilverCommentsRepository 



class PreprocessCommentsService:
    """
    Silver-layer builder:
      Bronze (JSONL) -> deterministic transform -> Silver (Parquet)
    """
    
    PREPROCESS_VERSION = "v1" # for further preprocessing
    
    SILVER_SCHEMA = pa.schema(
        [
            ("video_id", pa.string()),
            ("comment_id", pa.string()),
            ("author", pa.string()),
            ("published_at", pa.timestamp("us", tz="UTC")),
            ("like_count", pa.int64()),
            ("is_reply", pa.bool_()),
            ("text_raw", pa.string()),
            ("text_clean", pa.string()),
            ("preprocess_version", pa.string()),
            ("processed_at", pa.timestamp("us", tz="UTC")),
        ]
    )
    
    def __init__(
            self,
            bronze_repo: BronzeCommentsReader,
            silver_repo: ParquetSilverCommentsRepository, 
            text_preprocessor: TextPreprocessor,
    ) -> None:
        self._bronze_repo = bronze_repo
        self._silver_repo = silver_repo
        self._tp = text_preprocessor
        
    def run(self, video_id: str, *, overwrite: bool = True, batch_size: int = 5000) -> str:
        bronze_comments = self._bronze_repo.load(video_id)
        processed_at = datetime.now(timezone.utc)
        
        out_path = self._silver_repo.save(
            video_id,
            rows = self._iter_silver_rows(bronze_comments, processed_at=processed_at),
            schema = self.SILVER_SCHEMA,
            overwrite = overwrite,
            batch_size = 5000
        )
        return str(out_path)
        
            
    def _iter_silver_rows(self, comments: Iterable[Comment], *, processed_at: datetime) -> Iterable[dict]:
        rows: list[dict] = []
        
        for c in comments:
            yield self._comment_to_silver_row(c, processed_at=processed_at)
    
    def _comment_to_silver_row(self, c: Comment, *, processed_at: datetime) -> dict:
        raw = c.text
        cleaned = self._tp.clean(raw)
        
        published_at = c.published_at
        if published_at is None:
            published_at = datetime.fromtimestamp(0, tz=timezone.utc)
        elif published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)
        else:
            published_at = published_at.astimezone(timezone.utc)
            
        return {
            "video_id": c.video_id,
            "comment_id": c.comment_id,
            "author": c.author or "",
            "published_at": published_at,
            "like_count": int(c.like_count or 0),
            "is_reply": bool(c.is_reply),
            "text_raw": raw,
            "text_clean": cleaned,
            "preprocess_version": self.PREPROCESS_VERSION,
            "processed_at": datetime.now(timezone.utc),
        }