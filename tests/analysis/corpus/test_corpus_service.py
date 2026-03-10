from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yt_comments.analysis.corpus.service import CorpusService
from yt_comments.analysis.tfidf.models import TfidfConfig


def _write_silver_comments(path: Path, texts: list[str | None], preprocess_version) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    
    table = pa.table(
        {
            "text_clean": pa.array(texts, type=pa.string()),
            "preprocess_version": pa.array([preprocess_version] * len(texts), type=pa.string())
        }
    )
    pq.write_table(table, path)
    
def test_corpus_service_counts_features_once_per_video(tmp_path: Path) -> None:
    data_root = tmp_path / "data"

    _write_silver_comments(
        data_root / "silver" / "vid1" / "comments.parquet",
        ["amazing cat", "amazing cat"],
        "v1",
    )
    _write_silver_comments(
        data_root / "silver" / "vid2" / "comments.parquet",
        ["funny dog", "amazing dog"],
        "v1",
    )

    service = CorpusService(data_root=data_root)
    table = service.build(
        config=TfidfConfig(
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            min_ngram_df=1,
        )
    )

    got = {row.token: row.df_videos for row in table.tokens}

    assert table.artifact_version == "corpus_v1"
    assert table.preprocess_version == "v1"
    assert table.video_count == 2

    assert got["amazing"] == 2
    assert got["cat"] == 1
    assert got["dog"] == 1
    assert got["amazing cat"] == 1