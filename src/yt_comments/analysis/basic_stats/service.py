from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Iterable, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from yt_comments.analysis.basic_stats.models import BasicStats, BasicStatsConfig, TopToken



_TOKEN_RE = re.compile(r"[a-zA-Z0-9_']+")

class BasicStatsService:
    def __init__(self, *, preprocess_version: str = "v1") -> None:
        self._preprocess_version = preprocess_version
        
    def compute_for_video(
            self,
            *,
            video_id: str,
            silver_parquet_path: str,
            config: BasicStatsConfig,
            created_at_utc: Optional[datetime] = None,
            batch_size: int = 5000
    ) -> BasicStats:
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
                
                for tok in self._tokenize(comm, config):
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
            preprocess_version=self._preprocess_version,
            config_hash=self._hash_config(config),
            row_count=int(row_count),
            empty_text_count=int(empty_text_count),
            total_token_count=int(total_token_count),
            unique_token_count=int(unique_token_count),
            top_tokens=top_tokens,
        )
    
    @staticmethod
    def _tokenize(text: str, config: BasicStatsConfig) -> Iterable[str]:
        if config.lowercase:
            text = text.lower()
            
        for m in _TOKEN_RE.finditer(text): # finditer used since it returns a generator in comparison to findall
            tok = m.group(0) # returns the matched string
            
            if len(tok) < config.min_token_len:
                continue
            if config.drop_numeric_tokens and tok.isdigit():
                continue
            
            yield tok
            
    @staticmethod
    def _hash_config(config: BasicStatsConfig) -> str:
        payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode("utf-8") # dicts are not hashable, so need to convert to json string
        return  hashlib.sha256(payload).hexdigest()[:16]
            
                

            
            