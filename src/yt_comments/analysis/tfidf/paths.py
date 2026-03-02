from __future__ import annotations

from pathlib import Path


def tfidf_keywords_path(root: Path, video_id: str) -> Path:
    """
    Deterministic layout:
      data/gold/tfidf/<video_id>/keywords.parquet
    """
    return root / "gold" / "tfidf" / video_id / "keywords.parquet"   