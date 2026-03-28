from datetime import datetime, timezone
from pathlib import Path

import pytest

from yt_comments.analysis.basic_stats.models import TopToken
from yt_comments.analysis.channel_stats.models import ChannelTokenStats
from yt_comments.storage.gold_channel_token_stats_repository import ParquetChannelTokenStatsRepository


def test_channel_token_stats_repository_round_trip(tmp_path: Path):
    repo = ParquetChannelTokenStatsRepository(data_root=tmp_path / "data")
    
    stats_in = ChannelTokenStats(
        channel_id="chan123",
        video_ids=("v1", "v2", "v3"),
        created_at_utc=datetime(2026, 2, 17, 12, 0, 0, tzinfo=timezone.utc),
        preprocess_version="v1",
        config_hash="deadbeef",
        row_count=10,
        empty_text_count=2,
        total_token_count=100,
        unique_token_count=55,
        top_tokens=(TopToken("cat", 3), TopToken("dog", 2)),
    )
    
    repo.save(stats_in)
    stats_out = repo.load("chan123")
    
    assert stats_in == stats_out 
    
def test_channel_token_stats_repository_returns_error_loads_missing_file(tmp_path: Path):
    repo = ParquetChannelTokenStatsRepository(data_root=tmp_path / "data")
    
    with pytest.raises(FileNotFoundError):
        repo.load("chan123")