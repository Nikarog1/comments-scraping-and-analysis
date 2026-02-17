from __future__ import annotations

from pathlib import Path



def basic_stats_path(root: Path, video_id: str) -> Path:
    """
    Deterministic layout:
      data/gold/basic_stats/<video_id>/stats.parquet
    """
    return root / "gold" / "basic_stats" / video_id / "stats.parquet"