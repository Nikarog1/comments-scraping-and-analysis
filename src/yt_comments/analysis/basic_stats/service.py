from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Optional

import pyarrow.parquet as pq

from yt_comments.analysis.features import hash_config, tokenize, read_preprocess_version
from yt_comments.analysis.basic_stats.models import BasicStats, BasicStatsConfig, TopToken


class BasicStatsService:
    """Computes basic token-level statistics from preprocessed (Silver) comment data."""
        
    def compute_for_video(
            self,
            *,
            video_id: str,
            silver_parquet_path: str,
            config: BasicStatsConfig,
            created_at_utc: Optional[datetime] = None,
            batch_size: int = 5000
    ) -> BasicStats:
        """
        Compute basic descriptive statistics for a single video.

        Processes Silver parquet in batches, tokenizes text according to the provided
        config, and aggregates frequency-based metrics.

        Args:
            video_id: Target video identifier.
            silver_parquet_path: Path to Silver parquet file with cleaned texts.
            config: Tokenization and aggregation settings.
            created_at_utc: Timestamp for the artifact (defaults to current UTC).
            batch_size: Number of rows to process per batch.

        Returns:
            BasicStats artifact with counts and top tokens.
        """
        
        preprocess_version = read_preprocess_version(silver_parquet_path=silver_parquet_path)
        
        created_at_utc = created_at_utc or datetime.now(timezone.utc)
        if created_at_utc is None:
            raise ValueError("created_at_utc must be timezone-aware!")
        
        # using it to not depend on silver layer, i.e. to isolate this service
        # ParquetFile doesn't have __enter__, so can't use "with"
        pf = pq.ParquetFile(silver_parquet_path)
        
        row_count = 0
        empty_text_count = 0
        token_counts: Counter[str] = Counter()
        total_token_count = 0
        
        for batch in pf.iter_batches(batch_size=batch_size, columns=["text_clean"]):
            arr = batch.column(0) # transforming to py array 
            row_count += batch.num_rows 
            
            # converting comms to strings, store them in a list, and iterate through
            for comm in arr.to_pylist():
                if comm is None or str(comm).strip() == "":
                    empty_text_count += 1
                    continue
                
                for tok in tokenize(comm, config):
                    token_counts[tok] += 1
                    total_token_count += 1
        
        unique_token_count = len(token_counts)
        top_tokens = tuple(
            TopToken(token=tok, count=int(cnt))
            for tok, cnt in token_counts.most_common(config.top_n_tokens)
        )
        
        return BasicStats(
            video_id=video_id,
            silver_path=str(silver_parquet_path),
            created_at_utc=created_at_utc,
            preprocess_version=preprocess_version,
            config_hash=hash_config(config),
            row_count=int(row_count),
            empty_text_count=int(empty_text_count),
            total_token_count=int(total_token_count),
            unique_token_count=int(unique_token_count),
            top_tokens=top_tokens,
        )
            
                

            
            