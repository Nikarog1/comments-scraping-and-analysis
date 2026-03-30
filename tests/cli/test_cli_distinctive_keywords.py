from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from yt_comments.analysis.channel_tfidf.models import ChannelTfidfKeywords
from yt_comments.analysis.tfidf.models import TfidfConfig, TfidfKeyword, TfidfKeywords
from yt_comments.cli.main import main


def test_cli_distinctive_keywords_writes_gold_artifact(tmp_path: Path, capsys) -> None:
    channel_id = "chan123"
    video_id = "v1"  
    data_root = tmp_path / "data"

    config = TfidfConfig()

    video_tfidf = TfidfKeywords(
        video_id=video_id,
        created_at_utc=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        silver_path="some_path",
        preprocess_version="v1",
        artifact_version="v1",
        config_hash="deadbeef",
        row_count=10,
        empty_text_count=1,
        doc_count_non_empty=9,
        vocab_size=3,
        min_df_raw="2",
        max_df_raw="0.9",
        min_df_abs=2,
        max_df_abs=8,
        config=config,
        keywords=(
            TfidfKeyword(token="cat", score=0.42, idf=1.28, avg_tf=0.33, df=2),
        ),
    )

    channel_tfidf = ChannelTfidfKeywords(
        channel_id=channel_id,
        video_ids=(video_id, "v2"),
        created_at_utc=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        preprocess_version="v1",
        artifact_version="v1",
        config_hash="deadbeef",
        row_count=100,
        empty_text_count=5,
        doc_count_non_empty=95,
        vocab_size=10,
        min_df_raw="2",
        max_df_raw="0.9",
        min_df_abs=2,
        max_df_abs=90,
        config=config,
        keywords=(
            TfidfKeyword(token="cat", score=0.21, idf=1.10, avg_tf=0.19, df=10),
            TfidfKeyword(token="amazing", score=0.35, idf=1.28, avg_tf=0.27, df=8),
            TfidfKeyword(token="video", score=0.18, idf=1.69, avg_tf=0.11, df=1),
        ),
    )

    mock_repo_vid = Mock()
    mock_repo_vid.load.return_value = video_tfidf

    mock_repo_channel = Mock()
    mock_repo_channel.load.return_value = channel_tfidf

    with (
        patch("yt_comments.cli.main._load_channel_id_ref_mapping", return_value=channel_id),
        patch("yt_comments.cli.main.ParquetTfidfKeywordsRepository", return_value=mock_repo_vid),
        patch("yt_comments.cli.main.ParquetChannelTfidfKeywordsRepository", return_value=mock_repo_channel),
    ):
        exit_code = main(
            [
                "distinctive-keywords",
                channel_id,
                video_id,
                "--data-root",
                str(data_root),
            ]
        )

    assert exit_code == 0

    gold_path = data_root / "gold" / "distinctive_keywords" / channel_id / video_id / "keywords.parquet"
    assert gold_path.exists()

    out = capsys.readouterr().out
    assert f"channel_id: {channel_id}" in out
    assert f"video_id: {video_id}" in out
    assert "keyword_count:" in out
    assert "cat" in out
    assert "amazing" not in out
    assert "lift=" in out
    assert "video_score=" in out
    assert "channel_score=" in out
    assert "video_df=" in out
    assert "channel_df=" in out