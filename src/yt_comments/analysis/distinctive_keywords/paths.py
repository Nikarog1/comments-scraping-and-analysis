from __future__ import annotations

from pathlib import Path


def channel_distinctive_keywords_path(root: Path, channel_id: str, video_id: str) -> Path:
    """
    Deterministic layout:
      data/gold/distinctive_keywords/<channel_id>/<video_id>/keywords.parquet
    """
    return root / "gold" / "distinctive_keywords" / channel_id / video_id / "keywords.parquet"   