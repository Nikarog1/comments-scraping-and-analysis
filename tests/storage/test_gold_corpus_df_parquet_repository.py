from pathlib import Path

from yt_comments.analysis.corpus.models import CorpusDfTable, CorpusTokenStat
from yt_comments.storage.gold_corpus_df_parquet_repository import ParquetCorpusDfRepository



def test_corpus_repository_round_trip(tmp_path: Path) -> None:
    data_root = tmp_path / "data"

    repo = ParquetCorpusDfRepository(data_root)

    expected = CorpusDfTable(
        artifact_version="corpus_v1",
        preprocess_version="v1",
        config_hash="abc123",
        video_count=3,
        tokens=(
            CorpusTokenStat(token="amazing", df_videos=2),
            CorpusTokenStat(token="cat", df_videos=1),
            CorpusTokenStat(token="amazing cat", df_videos=1),
        ),
    )

    repo.save(expected)
    got = repo.load()

    assert got == expected