from __future__ import annotations

from pathlib import Path



def channel_token_stats_path(root: Path, channel_id: str) -> Path:
    """
    Deterministic layout:
      data/gold/channel_token_stats/<channel_id>/stats.parquet
    """
    return root / "gold" / "channel_token_stats" / channel_id / "stats.parquet"