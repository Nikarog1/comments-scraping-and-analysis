from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from yt_comments.ingestion.models import Comment 



class JSONLCommentsRepository:
    """
    Stores one JSON object per line (JSONL), one file per video_id:
      data/<video_id>.jsonl
    """
    
    def __init__(self, data_dir: Path | str = "data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def _path_for_video(self, video_id: str) -> Path:
        return self.data_dir / f"{video_id}.jsonl"
    
    def save(self, video_id: str, comments: Iterable[Comment], *, overwrite: bool = True) -> Path:
        """
        Save comments for a video_id to JSONL.

        overwrite=True means writing a fresh file each time.
        """
        # 2DO: add append too
        path = self._path_for_video(video_id)
        mode = "w" if overwrite else "a"
        
        with path.open(mode, encoding="utf-8") as f:
            for c in comments:
                record = self._comment_to_record(c)
                f.write(json.dumps(record, ensure_ascii=False))
                f.write("\n")
                
        return path
    
    def load(self, video_id: str) -> list[Comment]:
        """
        Load comments for a video_id from JSONL.
        Returns [] if the file does not exist.
        """
        path = self._path_for_video(video_id)
        if not path.exists():
            return []
        
        comments: list[Comment] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {path} at line {line_no}") from e
                try:
                    comments.append(self._record_to_comment(record))
                except (TypeError, ValueError, KeyError) as e:
                    raise ValueError(f"Invalid comment record in {path} at line {line_no}") from e
        
        return comments
    
    @staticmethod
    def _comment_to_record(comment: Comment) -> dict:
        record = asdict(comment)
        dt: datetime | None = record.get("published_at")
        record["published_at"] = dt.isoformat() if dt is not None else None
        return record
    
    @staticmethod
    def _record_to_comment(record: dict) -> Comment:
        published_at = record.get("published_at")
        if published_at is not None:
            record["published_at"] = datetime.fromisoformat(published_at)
        return Comment(**record)
            
        
