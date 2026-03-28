from __future__ import annotations

from pathlib import Path


def channel_tfidf_keywords_path(root: Path, channel_id: str) -> Path:
    """
    Deterministic layout:
      data/gold/channel_tfidf/<channel_id>/keywords.parquet
    """
    return root / "gold" / "channel_tfidf" / channel_id / "keywords.parquet"   