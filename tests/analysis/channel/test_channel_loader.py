from pathlib import Path
import pytest

import pyarrow as pa 
import pyarrow.parquet as pq

from yt_comments.analysis.channel.channel_loader import ChannelTextsLoader
from yt_comments.storage.silver_comments_repository import ParquetSilverCommentsRepository



def _write_silver_comments(path: Path, texts: list[str | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    
    table = pa.table(
        {
            "text_clean": pa.array(texts, type=pa.string())
        }
    )
    pq.write_table(table, path)
    
def test_channel_comms_loader(tmp_path: Path):
    data_root = tmp_path / "data"
    silver_root = data_root / "silver" / "comments"
    
    video_ids = ("v1", "v2")
    
    _write_silver_comments(
        silver_root / video_ids[0] / "comments.parquet",
        ["amazing", "cat"],
    )
    _write_silver_comments(
        silver_root / video_ids[1] / "comments.parquet",
        ["funny", "dog"],
    )
    
    repo = ParquetSilverCommentsRepository(silver_root)
    loader = ChannelTextsLoader(repo)
    
    result = list(loader.iter_texts(video_ids))
    
    assert result == ["amazing", "cat", "funny", "dog"]


def test_channel_texts_loader_missing_video(tmp_path: Path):
    data_root = tmp_path / "data"
    silver_root = data_root / "silver"

    video_ids = ("v1", "v2")

    # only v1 exists
    _write_silver_comments(
        silver_root / "v1" / "comments.parquet",
        ["hello"],
    )

    silver_repo = ParquetSilverCommentsRepository(silver_root)
    loader = ChannelTextsLoader(silver_repo)

    with pytest.raises(FileNotFoundError):
        list(loader.iter_texts(video_ids))
    
