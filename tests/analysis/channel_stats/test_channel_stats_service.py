from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yt_comments.analysis.basic_stats.models import BasicStatsConfig
from yt_comments.analysis.channel_stats.service import ChannelTokenStatsService
from yt_comments.storage.silver_comments_repository import ParquetSilverCommentsRepository



def _write_silver_comments(path: Path, texts: list[str | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    
    table = pa.table(
        {
            "text_clean": pa.array(texts, type=pa.string()),
            "preprocess_version": pa.array(["v1"] * len(texts), type=pa.string()),
        }
    )
    pq.write_table(table, path)
    

def test_channel_token_stats_service_returns_result(tmp_path: Path):  
    data_root = tmp_path / "data"
    silver_root = data_root / "silver" / "comments"
    
    video_ids = ("v1", "v2")
    
    _write_silver_comments(
        silver_root / video_ids[0] / "comments.parquet",
        ["amazing cat", ""],
    )
    _write_silver_comments(
        silver_root / video_ids[1] / "comments.parquet",
        ["funny cat", "hi"],
    )
    
    silver_repo = ParquetSilverCommentsRepository(silver_root)
    config = BasicStatsConfig() # keep defaults
    service = ChannelTokenStatsService()
    
    result = service.compute_for_channel(
        channel_id="chan123",
        video_ids=video_ids,
        silver_repo=silver_repo,
        config=config,
    )
    
    assert result.channel_id == "chan123"
    assert result.video_ids == video_ids
    assert result.row_count == 4
    assert result.empty_text_count == 1
    assert result.total_token_count == 4 # hi is a stopword
    assert result.unique_token_count == 3
    assert [(t.token, t.count) for t in result.top_tokens] == [("cat", 2), ("amazing", 1), ("funny", 1)]