from __future__ import annotations

from datetime import timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yt_comments.analysis.tfidf.models import TfidfConfig, TfidfKeyword, TfidfKeywords
from yt_comments.analysis.tfidf.paths import tfidf_keywords_path


def _require_utc(dt):
    if dt.tzinfo is None:
        raise ValueError("No timezone information is detected, check created_at_utc")
    # additional check: either dt is utc (offset 0)
    if dt.utcoffset() != timezone.utc.utcoffset(dt):
        raise ValueError("created_at_utc must be in UTC")
    
class ParquetTfidfKeywordsRepository:
    """
    Gold TF-IDF keywords repository: 1 row per video_id.

    Layout:
      <data_root>/gold/tfidf/<video_id>/keywords.parquet
    """
    def __init__(self, data_root: Path) -> None:
        self._data_root = data_root
        
    def save(self, keywords: TfidfKeywords) -> None:
        path = tfidf_keywords_path(self._data_root, keywords.video_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = self._to_table(keywords)
        pq.write_table(table, path)
        
    def load(self, video_id: str) -> TfidfKeywords:
        path = tfidf_keywords_path(self._data_root, video_id)
        if not path.exists():
            raise FileNotFoundError(f"Gold TF-IDF keywords are not found in {path.parent}")
        table = pq.read_table(path)
        return self._from_table(table)
    
    @classmethod
    def _to_table(cls, item: TfidfKeywords) -> pa.Table:
        _require_utc(item.created_at_utc)
        
        # keywords list<struct>
        kw_tokens: list[str] = []
        kw_scores: list[float] = []
        kw_avg_tfs: list[float] = []
        kw_idfs: list[float] = []
        kw_dfs: list[int] = []
        
        for kw in item.keywords:
            kw_tokens.append(kw.token)
            kw_scores.append(float(kw.score))
            kw_avg_tfs.append(float(kw.avg_tf))
            kw_idfs.append(float(kw.idf))
            kw_dfs.append(int(kw.df))
            
        keywords_struct_array = pa.StructArray.from_arrays(
            [
                pa.array(kw_tokens, type=pa.string()),
                pa.array(kw_scores, type=pa.float64()),
                pa.array(kw_avg_tfs, type=pa.float64()),
                pa.array(kw_idfs, type=pa.float64()),
                pa.array(kw_dfs, type=pa.int64()),
            ],
            fields=list(cls._schema().field("keywords").type.value_type),
        )
        
        keywords_list_array = pa.array([keywords_struct_array], type=pa.list_(keywords_struct_array.type))
        
        cfg = item.config
        data = {
            "video_id": [item.video_id],
            "created_at_utc": [item.created_at_utc],
            "silver_path": [item.silver_path],
            "preprocess_version": [item.preprocess_version],
            "config_hash": [item.config_hash],
            "row_count": [item.row_count],
            "empty_text_count": [item.empty_text_count],
            "doc_count_non_empty": [item.doc_count_non_empty],
            "vocab_size": [item.vocab_size],
            "top_k": [cfg.top_k],
            "min_token_len": [cfg.min_token_len],
            "drop_numeric_tokens": [cfg.drop_numeric_tokens],
            "drop_stopwords": [cfg.drop_stopwords],
            "lang": [cfg.lang],
            "lowercase": [cfg.lowercase],
            "min_df": [cfg.min_df],
            "max_df": [cfg.max_df],
            "min_df_raw": [str(cfg.min_df)],
            "max_df_raw": [str(cfg.max_df)],
            "min_df_abs": [item.min_df_abs],
            "max_df_abs": [item.max_df_abs],
            "tf_mode": [cfg.tf_mode],
            "idf_mode": [cfg.idf_mode],
            "keywords": keywords_list_array,
        }

        return pa.table(data, schema=cls._schema())
    
    @classmethod
    def _from_table(cls, table: pa.Table) -> TfidfKeywords:
        if table.num_rows != 1:
            raise ValueError(f"Expected exactly 1 row in stats table, got: {table.num_rows}")
        
        row = table.to_pylist()[0]
        
        _require_utc(row["created_at_utc"])
        
        kw_list = row["keywords"] or []
        keywords_list: list[TfidfKeyword] = [
            TfidfKeyword(
                token=kw["token"],
                score=float(kw["score"]),
                avg_tf=float(kw["avg_tf"]),
                idf=float(kw["idf"]),
                df=int(kw["df"]),
            )
            for kw in kw_list
        ]
        keywords: tuple[TfidfKeyword, ...] = tuple(keywords_list) # '...' repeat it 0 or more times
        
        cfg = TfidfConfig(
            top_k=int(row["top_k"]),
            min_token_len=int(row["min_token_len"]),
            drop_numeric_tokens=bool(row["drop_numeric_tokens"]),
            drop_stopwords=bool(row["drop_stopwords"]),
            lang=str(row["lang"]),
            lowercase=bool(row["lowercase"]),
            min_df=int(row["min_df_raw"]),
            max_df=float(row["max_df_raw"]),
            tf_mode=str(row["tf_mode"]),
            idf_mode=str(row["idf_mode"]),
        )
        
        return TfidfKeywords(
            video_id=str(row["video_id"]),
            created_at_utc=row["created_at_utc"],
            silver_path=str(row["silver_path"]),
            preprocess_version=str(row["preprocess_version"]),
            config_hash=str(row["config_hash"]),
            row_count=int(row["row_count"]),
            empty_text_count=int(row["empty_text_count"]),
            doc_count_non_empty=int(row["doc_count_non_empty"]),
            vocab_size=int(row["vocab_size"]),
            min_df_raw=str(row["min_df_raw"]),
            max_df_raw=str(row["max_df_raw"]),
            min_df_abs=int(row["min_df_abs"]),
            max_df_abs=int(row["max_df_abs"]),
            config=cfg,
            keywords=keywords,
        )

        
    @staticmethod
    def _schema() -> pa.Schema:
        keyword_struct = pa.struct(
            [
                ("token", pa.string()),
                ("score", pa.float64()),
                ("avg_tf", pa.float64()),
                ("idf", pa.float64()),
                ("df", pa.int64()),
            ]
        )
        return pa.schema(
            [
                ("video_id", pa.string()),
                ("created_at_utc", pa.timestamp("us", tz="UTC")),
                ("silver_path", pa.string()),
                ("preprocess_version", pa.string()),
                ("config_hash", pa.string()),
                ("row_count", pa.int64()),
                ("empty_text_count", pa.int64()),
                ("doc_count_non_empty", pa.int64()),
                ("vocab_size", pa.int64()),
                ("top_k", pa.int32()),
                ("min_token_len", pa.int32()),
                ("drop_numeric_tokens", pa.bool_()),
                ("drop_stopwords", pa.bool_()),
                ("lang", pa.string()),
                ("lowercase", pa.bool_()),
                ("min_df", pa.int64()),
                ("max_df", pa.float64()),
                ("min_df_raw", pa.string()),
                ("max_df_raw", pa.string()),
                ("min_df_abs", pa.int64()),
                ("max_df_abs", pa.int64()),
                ("tf_mode", pa.string()),
                ("idf_mode", pa.string()),
                ("keywords", pa.list_(keyword_struct)),
            ]
        )