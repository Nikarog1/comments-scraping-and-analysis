from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from yt_comments.ingestion.models import Comment
from yt_comments.preprocessing.preprocess_service import PreprocessCommentsService
from yt_comments.preprocessing.text_preprocessor import TextPreprocessor
from yt_comments.storage.bronze_comments_repository import JSONLCommentsRepository
from yt_comments.storage.silver_comments_repository import ParquetSilverCommentsRepository



def test_preprocess_streaming_writes_parquet(tmp_path: Path) -> None:
    bronze_repo = JSONLCommentsRepository(tmp_path / "bronze")
    silver_repo = ParquetSilverCommentsRepository(tmp_path / "silver")

    video_id = "abc123"
    bronze_repo.save(
        video_id,
        [
            Comment(
                video_id=video_id,
                comment_id="c1",
                text="Hello   WORLD! https://example.com",
                author="bob",
                like_count=None,
                published_at=datetime.fromisoformat("2026-01-01T10:00:00"),
                is_reply=False,
            )
        ],
    )

    svc = PreprocessCommentsService(
        bronze_repo=bronze_repo,
        silver_repo=silver_repo,
        text_preprocessor=TextPreprocessor(replace_urls_with="<URL>"),
    )

    out_path = Path(svc.run(video_id))

    assert out_path.exists()

    table = pq.read_table(out_path)
    df = table.to_pandas()

    print("COLUMNS:", df.columns.tolist())
    print("FIRST ROW KEYS:", table.schema.names)

    assert list(df.columns) == [
        "video_id",
        "comment_id",
        "author",
        "published_at",
        "like_count",
        "is_reply",
        "text_raw",
        "text_clean",
        "preprocess_version",
        "processed_at",
    ]

    assert df.loc[0, "text_clean"] == "hello world! <url>"
    assert df.loc[0, "author"] == "bob"
    assert df.loc[0, "like_count"] == 0
    assert df.loc[0, "preprocess_version"] == "v1"
    assert df.loc[0, "published_at"].to_pydatetime() == datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)