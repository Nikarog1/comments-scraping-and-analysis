from __future__ import annotations

import pytest

from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

from yt_comments.cli.main import main
from yt_comments.ingestion.models import Comment
from yt_comments.storage.bronze_comments_repository import JSONLCommentsRepository


def test_cli_scrape_prints_video_id(capsys) -> None:
    video_id = "dQw4w9WgXcQ" 
    
    exit_code = main(["scrape", "dQw4w9WgXcQ"]) # simulation of yt-comments scrape dQw4w9WgXcQ  
    assert exit_code == 0 # check the exit_code

    captured = capsys.readouterr()
    output = captured.out.strip()
    assert "Saved" in output # check that the json was saved
    assert video_id in output # check that video id was printed
    assert ".jsonl" in output # check that the file was created


def test_cli_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0 # check that calling --help returns SystemExit(0)
    
def test_cli_requires_command() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2 # check that no command returns 2
    
def test_cli_preprocess_creates_silver_parquet(tmp_path: Path) -> None:
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"

    # Seed Bronze
    video_id = "abc123"
    bronze_repo = JSONLCommentsRepository(bronze_dir)
    bronze_repo.save(
        video_id,
        [
            Comment(
                video_id=video_id,
                comment_id="c1",
                text="Hello WORLD! https://example.com",
                author="bob",
                like_count=1,
                published_at=datetime.fromisoformat("2026-01-01T10:00:00"),
                is_reply=False,
            )
        ],
    )

    rc = main(
        [
            "preprocess",
            video_id,
            "--bronze-dir",
            str(bronze_dir),
            "--silver-dir",
            str(silver_dir),
        ]
    )
    assert rc == 0

    out_path = silver_dir / video_id / "comments.parquet"
    assert out_path.exists()

    table = pq.ParquetFile(out_path).read()
    df = table.to_pandas()
    assert df.loc[0, "text_clean"] == "hello world! <url>"