from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq

from yt_comments.cli.main import main


def _write_silver_comments(silver_dir: Path, texts: list[str | None], preprocess_version: str) -> None:
    path = silver_dir / "comments.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    table = pa.table(
        {
            "text_clean": pa.array(texts, type=pa.string()),
            "preprocess_version": pa.array([preprocess_version] * len(texts))
        }
    )
    pq.write_table(table, path)
    
def test_cli_channel_tfidf_writes_gold_artifact(tmp_path: Path, capsys) -> None:
    channel_id = "chan123"
    video_ids = ("v1", "v2")
    data_root = tmp_path / "data"
    silver_dir = data_root / "silver" 
    
    _write_silver_comments(
        silver_dir / video_ids[0],
        [
            "cat cat",
            "dog",
        ],
        preprocess_version="v1",
    )

    _write_silver_comments(
        silver_dir / video_ids[1],
        [
            "cat dog",
            "bird",
        ],
        preprocess_version="v1",
    )
    
    class FakeSum:
        video_ids = ("v1", "v2")
    
    mock_summary_repo = Mock()
    mock_summary_repo.load_latest.return_value = FakeSum()

    with (
        patch("yt_comments.cli.main._load_channel_id_ref_mapping", return_value=channel_id),
        patch("yt_comments.cli.main.JSONChannelRunSummaryRepository", return_value=mock_summary_repo),
    ):
        exit_code = main(
            [
                "tfidf-channel",
                channel_id,
                "--data-root",
                str(data_root),
                "--silver-dir",
                str(silver_dir),
                "--lang",
                "en",
                "--keep-sentiment"
            ]
        )
    
    assert exit_code == 0
    
    gold_path = data_root / "gold" / "channel_tfidf" / channel_id / "keywords.parquet"
    assert gold_path.exists()
    
    out = capsys.readouterr().out
    assert f"channel_id: {channel_id}" in out
    assert "top_keywords:" in out
    assert "cat" in out
    assert "dog" in out
    assert "score=" in out
    assert "df=" in out
    
def test_cli_tfidf_channel_invalid_ngram_range_returns_2(tmp_path: Path, capsys) -> None:
    channel_id = "chan123"
    data_root = tmp_path / "data"

    class FakeSum:
        video_ids = ("v1", "v2")

    mock_summary_repo = Mock()
    mock_summary_repo.load_latest.return_value = FakeSum()

    with (
        patch("yt_comments.cli.main._load_channel_id_ref_mapping", return_value=channel_id),
        patch("yt_comments.cli.main.JSONChannelRunSummaryRepository", return_value=mock_summary_repo),
    ):
        exit_code = main(
            [
                "tfidf-channel",
                channel_id,
                "--data-root",
                str(data_root),
                "--ngram-min",
                "2",
                "--ngram-max",
                "1",
            ]
        )

    assert exit_code == 2

    out = capsys.readouterr().out
    assert out == ""

    gold_path = data_root / "gold" / "channel_tfidf" / channel_id / "keywords.parquet"
    assert not gold_path.exists()