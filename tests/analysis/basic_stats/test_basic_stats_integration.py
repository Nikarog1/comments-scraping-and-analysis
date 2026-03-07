from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from yt_comments.analysis.basic_stats.models import BasicStatsConfig
from yt_comments.analysis.basic_stats.service import BasicStatsService
from yt_comments.storage.gold_basic_stats_parquet_repository import ParquetBasicStatsRepository



def test_basic_stats_service_to_repo_round_trip(tmp_path):
    silver_path = tmp_path / "data" / "silver" / "vid1" / "comments.parquet"
    silver_path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pydict(
        {
            "text_clean": [
                "hello world",
                "hello",
                "",
                None,
                "world world 123",
            ]
        }
    )
    pq.write_table(table, silver_path)

    svc = BasicStatsService(preprocess_version="v1")
    cfg = BasicStatsConfig(top_n_tokens=10, min_token_len=2, drop_numeric_tokens=True)

    stats = svc.compute_for_video(
        video_id="vid1",
        silver_parquet_path=str(silver_path),
        config=cfg,
        created_at_utc=datetime(2026, 2, 17, 12, 0, 0, tzinfo=timezone.utc),
        batch_size=2,  # force multiple batches
    )

    repo = ParquetBasicStatsRepository(data_root=tmp_path / "data")
    repo.save(stats)

    loaded = repo.load("vid1")
    assert loaded == stats

