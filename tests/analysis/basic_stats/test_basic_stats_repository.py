from __future__ import annotations

from datetime import datetime, timezone

from yt_comments.analysis.basic_stats.models import BasicStats, TopToken
from yt_comments.storage.gold_basic_stats_parquet_repository import ParquetBasicStatsRepository



def test_parquet_basic_stats_repository_round_trip(tmp_path):
    repo = ParquetBasicStatsRepository(data_root=tmp_path / "data")

    stats_in = BasicStats(
        video_id="abc123",
        silver_path="data/silver/abc123/comments.parquet",
        created_at_utc=datetime(2026, 2, 17, 12, 0, 0, tzinfo=timezone.utc),
        preprocess_version="v1",
        config_hash="deadbeef",
        row_count=10,
        empty_text_count=2,
        total_token_count=100,
        unique_token_count=55,
        top_tokens=(TopToken("test", 3), TopToken("eat", 2)),
    )

    repo.save(stats_in)
    stats_out = repo.load("abc123")

    assert stats_out == stats_in