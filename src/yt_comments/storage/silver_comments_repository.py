from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq



class ParquetSilverCommentsRepository:
    """
    Silver layer repo: streaming parquet writer
    
    Layout:
      data/silver/comments/video_id=<id>/comments.parquet
    """
    
    def __init__(self, base_dir: Path | str = "data/silver/comments") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def _dir_for_video(self, video_id: str) -> Path:
        return self.base_dir / f"video_id={video_id}"
    
    def _path_for_comments(self, video_id: str) -> Path:
        return self._dir_for_video(video_id) / f"comments.parquet"
    
    def save(
            self, 
            video_id: str, 
            rows: Iterable[dict], 
            *, 
            schema: pa.Schema,
            overwrite: bool = True,
            batch_size: int = 5000,
        ) -> Path:
        
        out_dir = self._dir_for_video(video_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        path = self._path_for_comments(video_id)
        if path.exists() and not overwrite:
            raise ValueError(f"Refusing to overwrite already existing file in {path}")
        
        # delete old files
        if path.exists() and overwrite:
            path.unlink()
            
        with pq.ParquetWriter(path, schema=schema) as w:
            buffer: list[dict] = []
            for row in rows:
                buffer.append(row)
                if len(buffer) >= batch_size:
                    w.write_table(pa.Table.from_pylist(buffer, schema=schema))
                    buffer.clear()
                    
            if buffer:
                w.write_table(pa.Table.from_pylist(buffer, schema=schema))
        
        return path
    
    def load(self, video_id: str) -> pa.Table:
        path = self._path_for_comments(video_id)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found for video id = {video_id}"
            )
        
        return pq.read_table(path)