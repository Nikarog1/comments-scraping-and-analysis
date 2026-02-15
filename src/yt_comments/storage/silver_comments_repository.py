from __future__ import annotations

from pathlib import Path

import pandas as pd



class ParquetSilverCommentsRepository:
    """
    Silver layer repo: analysis ready, columnar storage
    """
    
    def __init__(self, base_dir: Path | str = "data/silver/comments") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def _dir_for_video(self, video_id: str) -> Path:
        return self.base_dir / f"video_id = {video_id}"
    
    def _path_for_comments(self, video_id: str) -> Path:
        return self._dir_for_video(video_id) / f"comments.parquet"
    
    def save(self, video_id: str, df: pd.DataFrame, *, overwrite: bool = True) -> Path:
        out_dir = self._dir_for_video(video_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        path = self._path_for_comments(video_id)
        if path.exists() and not overwrite:
            raise ValueError(f"Refusing to overwrite already existing file in {path}")
        
        df.to_parquet(path, index=False)
        
        return path
    
    def load(self, video_id: str) -> pd.DataFrame:
        path = self._path_for_comments(video_id)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found for video id = {video_id}"
            )
        
        return pd.read_parquet(path)