from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yt_comments.cli.main import main
from yt_comments.storage.gold_tfidf_keywords_parquet_repository import ParquetTfidfKeywordsRepository


def _write_silver_comments(path: Path, texts: list[str | None], preprocess_version) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    
    table = pa.table(
        {
            "text_clean": pa.array(texts, type=pa.string()),
            "preprocess_version": pa.array([preprocess_version]*len(texts), type=pa.string())
        }
    )
    pq.write_table(table, path)

def test_cli_tfidf_bigrams(tmp_path: Path):
    data_root = tmp_path / "data"
    video_id = "vid1"

    silver_path = data_root / "silver" / video_id / "comments.parquet"
    _write_silver_comments(
        silver_path,
        [
            "amazing cat",
            "amazing cat",
            "funny dog",
            "cool vid"
        ],
        "v1"
    )

    main(
        [
            "tfidf",
            video_id,
            "--data-root",
            str(data_root),
            "--keep-sentiment",
            "--ngram-min",
            "1",
            "--ngram-max",
            "2",
            "--min-ngram-df",
            "2"
        ]
    )
    
    repo = ParquetTfidfKeywordsRepository(data_root)
    table = repo.load(video_id)

    tokens = {kw.token for kw in table.keywords}

    assert tokens 
    assert "amazing cat" in tokens
    assert "funny dog" not in tokens
    assert "cat" in tokens