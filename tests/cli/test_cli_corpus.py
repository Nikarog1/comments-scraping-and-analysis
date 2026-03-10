from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yt_comments.cli.main import main
from yt_comments.storage.gold_corpus_df_parquet_repository import ParquetCorpusDfRepository


def _write_silver_comments(
    path: Path,
    texts: list[str | None],
    preprocess_version: str = "v1",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.table(
        {
            "text_clean": pa.array(texts, type=pa.string()),
            "preprocess_version": pa.array([preprocess_version] * len(texts), type=pa.string()),
        }
    )

    pq.write_table(table, path)


def test_cli_corpus_builds_artifact(tmp_path: Path) -> None:
    data_root = tmp_path / "data"

    _write_silver_comments(
        data_root / "silver" / "vid1" / "comments.parquet",
        ["amazing cat", "amazing cat"],
    )

    _write_silver_comments(
        data_root / "silver" / "vid2" / "comments.parquet",
        ["funny dog", "amazing dog"],
    )

    main(
        [
            "corpus",
            "--data-root",
            str(data_root),
        ]
    )

    repo = ParquetCorpusDfRepository(data_root)
    table = repo.load()

    tokens = {row.token: row.df_videos for row in table.tokens}

    assert table.artifact_version == "corpus_v1"
    assert table.preprocess_version == "v1"
    assert table.video_count == 2

    assert tokens["amazing"] == 2
    assert tokens["cat"] == 1
    assert tokens["dog"] == 1