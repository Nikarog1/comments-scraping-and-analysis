from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yt_comments.cli.main import main


def _write_silver_comments(path: Path, texts: list[str | None], preprocess_version) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    
    table = pa.table(
        {
            "text_clean": pa.array(texts, type=pa.string()),
            "preprocess_version": pa.array([preprocess_version]*len(texts), type=pa.string())
        }
    )
    pq.write_table(table, path)
    
def test_cli_tfidf_writes_gold_artifact(tmp_path: Path, capsys) -> None:
    data_root = tmp_path / "data"
    video_id = "vid1"

    silver_path = data_root / "silver" / video_id / "comments.parquet"
    _write_silver_comments(
        silver_path,
        [
            "i love this cat",
            "this cat is amazing",
            "amazing video love it",
        ],
        "v1"
    )

    exit_code = main(
        [
            "tfidf",
            video_id,
            "--data-root",
            str(data_root),
            "--top-k",
            "10",
            "--min-token-len",
            "2",
           "--min-df",
            "1",
            "--max-df",
            "1.0",
            "--lang",
            "en",
            "--batch-size",
            "2",
        ]
    )
    
    assert exit_code == 0
    
    gold_path = data_root / "gold" / "tfidf" / video_id / "keywords.parquet"
    assert gold_path.exists()
    
    out = capsys.readouterr().out
    assert f"video_id: {video_id}" in out
    assert "top_keywords:" in out
    assert "cat" in out
    assert "amazing" in out
    assert "score=" in out
    assert "df=" in out