from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yt_comments.analysis.features import hash_config
from yt_comments.analysis.corpus.models import CorpusDfTable, CorpusTokenStat
from yt_comments.analysis.tfidf.models import TfidfConfig
from yt_comments.analysis.tfidf.service import TfidfService


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
    
def test_tfidf_service_uses_global_corpus_idf(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    silver_path = data_root / "silver" / "vid1" / "comments.parquet"
    
    _write_silver_comments(
        silver_path,
        [
            "amazing cat",
            "amazing cat",
        ]
    )
    
    config = TfidfConfig(        
        top_k=10,
        min_df=1,
        max_df=1.0,
        ngram_range=(1,2),
        min_ngram_df=1
    )
    
    corpus = CorpusDfTable(
        artifact_version="corpus_v1",
        preprocess_version="v1",
        config_hash=hash_config(config),
        video_count=10,
        tokens=(
            CorpusTokenStat("amazing", 7),
            CorpusTokenStat("cat", 5),
            CorpusTokenStat("amazing cat", 2),
        )
    )
    
    service = TfidfService()
    result = service.compute_for_video(
        video_id="vid1",
        silver_parquet_path=silver_path,
        config=config,
        global_corpus=corpus,
        unfilter_sentiment=False,
    )
    
    tokens = {row.token: row for row in result.keywords}
    
    assert result.artifact_version == "tfidf_v3"
    
    assert tokens["amazing"].df == 7
    assert tokens["cat"].df == 5
    
    assert "amazing cat" in tokens