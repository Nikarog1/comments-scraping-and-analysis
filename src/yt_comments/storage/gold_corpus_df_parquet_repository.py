from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yt_comments.analysis.corpus.models import CorpusDfTable, CorpusTokenStat
from yt_comments.analysis.corpus.paths import tfidf_corpus_df_path


class ParquetCorpusDfRepository:
    """
    Gold v3 TF-IDF corpus repository.

    Layout:
      <data_root>/gold/corpus_df/corpus.parquet
    """
    _SCHEMA = pa.schema(
        [
            pa.field("artifact_version", pa.string(), nullable=False),
            pa.field("preprocess_version", pa.string(), nullable=False),
            pa.field("config_hash", pa.string(), nullable=False),
            pa.field("video_count", pa.int64(), nullable=False),
            pa.field("tokens", pa.list_(pa.string()), nullable=False),
            pa.field("df_videos", pa.list_(pa.int64()), nullable=False),
        ]
    )
    
    def __init__(self, data_root: Path) -> None:
        self._data_root = data_root
        
    def save(self, corpus: CorpusDfTable) -> None:
        path = tfidf_corpus_df_path(self._data_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        table = self._to_table(corpus)
        
        pq.write_table(table, path)
        
    def load(self) -> CorpusDfTable:
        path = tfidf_corpus_df_path(self._data_root)
        table = pq.read_table(path, schema=self._SCHEMA)
        return self._from_table(table)
        
    def _to_table(self, corpus: CorpusDfTable) -> pa.Table:
        tokens = [row.token for row in corpus.tokens]
        dfs = [row.df_videos for row in corpus.tokens]

        return pa.Table.from_pylist(
            [
                {
                    "artifact_version": corpus.artifact_version,
                    "preprocess_version": corpus.preprocess_version,
                    "config_hash": corpus.config_hash,
                    "video_count": corpus.video_count,
                    "tokens": tokens,
                    "df_videos": dfs,
                }
            ],
            schema=self._SCHEMA,
        )

    def _from_table(self, table: pa.Table) -> CorpusDfTable:
        rows = table.to_pylist()
        if len(rows) != 1:
            raise ValueError(
                f"Expected exactly 1 row in corpus artifact, got {len(rows)}"
            )

        row = rows[0]
        tokens = row["tokens"]
        dfs = row["df_videos"]

        if len(tokens) != len(dfs):
            raise ValueError(
                "Invalid corpus artifact: tokens and df_videos length mismatch"
            )

        return CorpusDfTable(
            artifact_version=row["artifact_version"],
            preprocess_version=row["preprocess_version"],
            config_hash=row["config_hash"],
            video_count=row["video_count"],
            tokens=tuple(
                CorpusTokenStat(token=token, df_videos=df)
                for token, df in zip(tokens, dfs)
            ),
        )