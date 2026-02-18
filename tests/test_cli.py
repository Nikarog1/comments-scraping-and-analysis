from __future__ import annotations

import pytest

from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yt_comments.cli.main import main
from yt_comments.ingestion.models import Comment
from yt_comments.storage.bronze_comments_repository import JSONLCommentsRepository
from yt_comments.storage.gold_basic_stats_parquet_repository import ParquetBasicStatsRepository



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
    

def test_cli_stats_writes_gold_artifact(tmp_path, capsys):
    data_root = tmp_path / "data"
    video_id = "vid1"

    silver_path = data_root / "silver" / video_id / "comments.parquet"
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pydict({"text_clean": ["the the hello", "world", None, "and you", "cool 123"]}),
        silver_path,
    )

    rc = main(
        [
            "stats",
            video_id,
            "--data-root",
            str(data_root),
            "--top-n",
            "5",
            "--lang",
            "en",
        ]
    )
    print(data_root)
    assert rc == 0
    
    expected = data_root / "gold" / "basic_stats" / video_id / "stats.parquet"
    local_written = Path("data") / "gold" / "basic_stats" / video_id / "stats.parquet"

    # Debug prints (pytest will show on failure)
    print("expected:", expected)
    print("exists expected:", expected.exists())
    print("local:", local_written)
    print("exists local:", local_written.exists())

    # Also show what was actually created under tmp data_root
    print("tmp tree:", [p.relative_to(data_root) for p in data_root.rglob("*")])

    repo = ParquetBasicStatsRepository(data_root=data_root)
    stats = repo.load(video_id)

    assert stats.video_id == video_id
    assert stats.row_count == 5
    assert stats.empty_text_count == 1  # None
    assert sorted((t.token, t.count) for t in stats.top_tokens) == [("cool", 1), ("hello", 1), ("world", 1)]

    # sanity-check CLI printed something
    out = capsys.readouterr().out
    assert "video_id" in out