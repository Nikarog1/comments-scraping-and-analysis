from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from yt_comments.analysis.basic_stats.models import BasicStats, TopToken
from yt_comments.analysis.basic_stats.paths import basic_stats_path



class ParquetBasicStatsRepository:
    """
    Gold v1 storage: parquet per video
    
    Layout:
      data/gold/basic_stats/<video_id>/stats.parquet
    """
    
    def __init__(self, data_root: Path) -> None:
        self._data_root = data_root
        
    def save(self, stats: BasicStats) -> None:
        path = basic_stats_path(self._data_root, stats.video_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        table = self._to_table(stats)
        pq.write_table(table, path)
    
    def load(self, video_id: str) -> BasicStats:
        path = basic_stats_path(self._data_root, video_id)
        if not path.exists():
            raise FileNotFoundError(f"Gold basic stats are not found in {path}")
        
        table = pq.read_table(path)
        return self._from_table(table)
    
    @staticmethod
    def _to_table(stats: BasicStats) -> pa.Table:
        top_tokens_struct = pa.struct([("token", pa.string()), ("count", pa.int64())])
        top_tokens_array = pa.array(
            [[{"token": t.token, "count": int(t.count)} for t in stats.top_tokens]],
            type=pa.list_(top_tokens_struct)
        )
        
        # to early detect problems with timezones 
        created_at = stats.created_at_utc
        if created_at.tzinfo is None:
            raise ValueError("No timezone information is detected, check created_at_utc")
        
        schema = pa.schema(
            [
                ("video_id", pa.string()),
                ("silver_path", pa.string()),
                ("created_at_utc", pa.timestamp("us", tz="UTC")),
                ("preprocess_version", pa.string()),
                ("config_hash", pa.string()),
                ("row_count", pa.int64()),
                ("empty_text_count", pa.int64()),
                ("total_token_count", pa.int64()),
                ("unique_token_count", pa.int64()),
                ("top_tokens", pa.list_(top_tokens_struct)),
            ]
        )
        
        data = {
            "video_id": [stats.video_id],
            "silver_path": [stats.silver_path],
            "created_at_utc": [created_at],
            "preprocess_version": [stats.preprocess_version],
            "config_hash": [stats.config_hash],
            "row_count": [int(stats.row_count)],
            "empty_text_count": [int(stats.empty_text_count)],
            "total_token_count": [int(stats.total_token_count)],
            "unique_token_count": [int(stats.unique_token_count)],
            "top_tokens": top_tokens_array,
        }
        
        return pa.Table.from_pydict(data, schema=schema)
    
    @staticmethod
    def _from_table(table: pa.Table) -> BasicStats:
        if table.num_rows != 1:
            raise ValueError(f"Expected exactly 1 row in stats table, got: {table.num_rows}")
        
        row = table.to_pylist()[0]
        
        # to prevent silent time zone drift
        created_at = row["created_at_utc"]
        if isinstance(created_at, datetime) and created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
            
        top_tokens_py = row.get("top_tokens") or []
        top_tokens = tuple(TopToken(token=t["token"], count=int(t["count"])) for t in top_tokens_py)
        
        return BasicStats(
            video_id=row["video_id"],
            silver_path=row["silver_path"],
            created_at_utc=created_at,
            preprocess_version=row["preprocess_version"],
            config_hash=row["config_hash"],
            row_count=int(row["row_count"]),
            empty_text_count=int(row["empty_text_count"]),
            total_token_count=int(row["total_token_count"]),
            unique_token_count=int(row["unique_token_count"]),
            top_tokens=top_tokens,
        )
        
            
        