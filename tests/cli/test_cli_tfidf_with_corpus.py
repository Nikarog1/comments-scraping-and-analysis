from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yt_comments.analysis.corpus.models import CorpusDfTable, CorpusTokenStat
from yt_comments.analysis.features import hash_config
from yt_comments.analysis.tfidf.models import TfidfConfig

from yt_comments.cli.main import main

from yt_comments.nlp.stopwords import STOPWORDS

from yt_comments.storage.gold_corpus_df_parquet_repository import ParquetCorpusDfRepository
from yt_comments.storage.gold_tfidf_keywords_parquet_repository import ParquetTfidfKeywordsRepository


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


def test_cli_tfidf_uses_global_corpus(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    video_id = "vid1"
    silver_root = data_root / "silver" / video_id / "comments.parquet"
    
    _write_silver_comments(
        silver_root,
        [
            "amazing cat",
            "amazing cat",
        ]
    )
    
    config = TfidfConfig(
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        min_ngram_df=1,
        stopwords_lang="en",
        stopwords_hash=str(hash_config(sorted(STOPWORDS["en"])))
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
        ),
    )
    
    ParquetCorpusDfRepository(data_root=data_root).save(corpus)

    main(
        [
            "tfidf",
            video_id,
            "--data-root",
            str(data_root),
            "--use-corpus",
            "--lang",
            "en",
            "--ngram-min",
            "1",
            "--ngram-max",
            "2",
            "--min-df",
            "1",
            "--max-df",
            "1.0",
            "--min-ngram-df",
            "1"
            
        ]
    )
    
    table = ParquetTfidfKeywordsRepository(data_root=data_root).load(video_id=video_id)
    
    tokens = {row.token: row for row in table.keywords}
    
    assert table.artifact_version == "tfidf_v3"
    assert table.preprocess_version == "v1"
    
    assert tokens["cat"].df == 5
    assert tokens["amazing cat"].df == 2