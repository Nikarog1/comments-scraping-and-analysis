from __future__ import annotations

from datetime import timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yt_comments.analysis.distinctive_keywords.models import DistinctiveKeyword, DistinctiveKeywords
from yt_comments.analysis.distinctive_keywords.paths import channel_distinctive_keywords_path
from yt_comments.analysis.tfidf.models import TfidfConfig, TfidfKeyword



def _require_utc(dt) -> None:
    if dt.tzinfo is None:
        raise ValueError("No timezone information is detected, check created_at_utc")
    # additional check: either dt is utc (offset 0)
    if dt.utcoffset() != timezone.utc.utcoffset(dt):
        raise ValueError("created_at_utc must be in UTC")
    

class ParquetDistinctiveKeywordsRepository:
    """
    Gold TF-IDF distinctive keywords repository: 1 row per video.

    Layout:
      <data_root>/gold/distinctive_keywords/<channel_id>/<video_id>/keywords.parquet
    """
    def __init__(self, data_root: str | Path = "data") -> None:
        self._data_root = Path(data_root)
        
    def save(self, keywords: DistinctiveKeywords) -> None:
        path = channel_distinctive_keywords_path(self._data_root, keywords.channel_id, keywords.video_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = self._to_table(keywords)
        pq.write_table(table, path)
        
    def load(self, channel_id: str, video_id: str) -> DistinctiveKeywords:
        path = channel_distinctive_keywords_path(self._data_root, channel_id, video_id)
        if not path.exists():
            raise FileNotFoundError(f"Distinctive keywords not found for channel={channel_id}, video={video_id}")
        table = pq.read_table(path)
        return self._from_table(table)
    
    @classmethod
    def _to_table(cls, item: DistinctiveKeywords) -> pa.Table:
        _require_utc(item.created_at_utc)

        kw_tokens = []
        kw_video_scores = []
        kw_channel_scores = []
        kw_lifts = []
        kw_video_dfs = []
        kw_channel_dfs = []

        for kw in item.keywords:
            kw_tokens.append(kw.token)
            kw_video_scores.append(float(kw.video_score))
            kw_channel_scores.append(float(kw.channel_score))
            kw_lifts.append(float(kw.lift))
            kw_video_dfs.append(int(kw.video_df))
            kw_channel_dfs.append(int(kw.channel_df))

        keyword_struct = pa.StructArray.from_arrays(
            [
                pa.array(kw_tokens, type=pa.string()),
                pa.array(kw_video_scores, type=pa.float64()),
                pa.array(kw_channel_scores, type=pa.float64()),
                pa.array(kw_lifts, type=pa.float64()),
                pa.array(kw_video_dfs, type=pa.int64()),
                pa.array(kw_channel_dfs, type=pa.int64()),
            ],
            fields=list(cls._schema().field("keywords").type.value_type),
        )

        keywords_array = pa.array(
            [keyword_struct],
            type=pa.list_(keyword_struct.type),
        )

        cfg = item.config

        data = {
            "channel_id": [item.channel_id],
            "video_id": [item.video_id],
            "created_at_utc": [item.created_at_utc],
            "preprocess_version": [item.preprocess_version],
            "artifact_version": [item.artifact_version],
            "config_hash": [item.config_hash],
            "keyword_count": [item.keyword_count],
            "top_k": [cfg.top_k],
            "min_token_len": [cfg.min_token_len],
            "drop_numeric_tokens": [cfg.drop_numeric_tokens],
            "drop_stopwords": [cfg.drop_stopwords],
            "stopwords_lang": [cfg.stopwords_lang],
            "stopwords_hash": [cfg.stopwords_hash],
            "normalization": [cfg.normalization],
            "lowercase": [cfg.lowercase],
            "min_df": [cfg.min_df],
            "max_df": [cfg.max_df],
            "tf_mode": [cfg.tf_mode],
            "idf_mode": [cfg.idf_mode],
            "keywords": keywords_array,
        }

        return pa.table(data, schema=cls._schema())

    @classmethod
    def _from_table(cls, table: pa.Table) -> DistinctiveKeywords:
        if table.num_rows != 1:
            raise ValueError(f"Expected 1 row, got {table.num_rows}")

        row = table.to_pylist()[0]

        _require_utc(row["created_at_utc"])

        keywords = tuple(
            DistinctiveKeyword(
                token=kw["token"],
                video_score=float(kw["video_score"]),
                channel_score=float(kw["channel_score"]),
                lift=float(kw["lift"]),
                video_df=int(kw["video_df"]),
                channel_df=int(kw["channel_df"]),
            )
            for kw in (row["keywords"] or [])
        )

        cfg = TfidfConfig(
            top_k=int(row["top_k"]),
            min_token_len=int(row["min_token_len"]),
            drop_numeric_tokens=bool(row["drop_numeric_tokens"]),
            drop_stopwords=bool(row["drop_stopwords"]),
            stopwords_lang=str(row["stopwords_lang"]),
            stopwords_hash=str(row["stopwords_hash"]),
            normalization=str(row["normalization"]),
            lowercase=bool(row["lowercase"]),
            min_df=row["min_df"],
            max_df=row["max_df"],
            tf_mode=str(row["tf_mode"]),
            idf_mode=str(row["idf_mode"]),
        )

        return DistinctiveKeywords(
            channel_id=str(row["channel_id"]),
            video_id=str(row["video_id"]),
            created_at_utc=row["created_at_utc"],
            preprocess_version=str(row["preprocess_version"]),
            artifact_version=str(row["artifact_version"]),
            config_hash=str(row["config_hash"]),
            config=cfg,
            keyword_count=int(row["keyword_count"]),
            keywords=keywords,
        )

    @staticmethod
    def _schema() -> pa.Schema:
        keyword_struct = pa.struct(
            [
                ("token", pa.string()),
                ("video_score", pa.float64()),
                ("channel_score", pa.float64()),
                ("lift", pa.float64()),
                ("video_df", pa.int64()),
                ("channel_df", pa.int64()),
            ]
        )

        return pa.schema(
            [
                ("channel_id", pa.string()),
                ("video_id", pa.string()),
                ("created_at_utc", pa.timestamp("us", tz="UTC")),
                ("preprocess_version", pa.string()),
                ("artifact_version", pa.string()),
                ("config_hash", pa.string()),
                ("keyword_count", pa.int64()),
                ("top_k", pa.int32()),
                ("min_token_len", pa.int32()),
                ("drop_numeric_tokens", pa.bool_()),
                ("drop_stopwords", pa.bool_()),
                ("stopwords_lang", pa.string()),
                ("stopwords_hash", pa.string()),
                ("normalization", pa.string()),
                ("lowercase", pa.bool_()),
                ("min_df", pa.float64()),
                ("max_df", pa.float64()),
                ("tf_mode", pa.string()),
                ("idf_mode", pa.string()),
                ("keywords", pa.list_(keyword_struct)),
            ]
        )