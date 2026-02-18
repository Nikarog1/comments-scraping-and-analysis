from __future__ import annotations

from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from yt_comments.analysis.basic_stats.models import BasicStatsConfig
from yt_comments.analysis.basic_stats.service import BasicStatsService


def test_basic_stats_service_computes_counts(tmp_path):
    silver_path = tmp_path / "comments.parquet"

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
    config = BasicStatsConfig(top_n_tokens=10, min_token_len=2, drop_numeric_tokens=True)

    stats = svc.compute_for_video(
        video_id="vid1",
        silver_parquet_path=str(silver_path),
        config=config,
        created_at_utc=datetime(2026, 2, 17, 12, 0, 0, tzinfo=timezone.utc),
    )

    assert stats.video_id == "vid1"
    assert stats.row_count == 5
    assert stats.empty_text_count == 2  # "" and None

    assert stats.total_token_count == 5
    assert stats.unique_token_count == 2
    assert [(t.token, t.count) for t in stats.top_tokens] == [("world", 3), ("hello", 2)]