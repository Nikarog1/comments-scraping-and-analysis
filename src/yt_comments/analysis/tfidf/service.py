import hashlib
import json
import math
import re
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Iterable, Optional

import pyarrow.parquet as pq

from yt_comments.analysis.tfidf.accumulator import TfidfAccumulator
from yt_comments.analysis.tfidf.models import TfidfConfig, TfidfKeyword, TfidfKeywords
from yt_comments.nlp.stopwords import get_stopwords


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_']+")

class TfidfService:
    """
    Gold v2 TF-IDF keywords service (per-video corpus).

    Document unit: one comment
    Corpus: all comments of a single video (Silver parquet)
    """
    
    def __init__(self, *, preprocess_version: str = "v1") -> None:
        self._preprocess_version = preprocess_version
        
    def compute_for_video(
            self,
            *,
            video_id: str,
            silver_parquet_path: str,
            config: TfidfConfig,
            created_at_utc: Optional[datetime] = None,
            batch_size: int = 5000,
    ) -> TfidfKeywords:
        
        created_at_utc = created_at_utc or datetime.now(timezone.utc)
        if created_at_utc.tzinfo is None:
            raise ValueError("created_at_utc must be timezone-aware!")
        if created_at_utc.utcoffset() != timezone.utc.utcoffset(created_at_utc):
            raise ValueError("created_at_utc must be in UTC")
        
        acc = TfidfAccumulator()
        
        pf = pq.ParquetFile(silver_parquet_path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=["text_clean"]):
            arr = batch.column(0)
            
            for comm in arr.to_pylist():
                tokens = []
                if comm is not None and str(comm).strip() != "":
                    tokens = list(self._tokenize(str(comm), config))
                acc.add_document(tokens)
                
        N = acc.doc_count_non_empty
        min_df_abs, max_df_abs = self._resolve_df_thresholds(
            min_df=config.min_df,
            max_df=config.max_df,
            N=N
        )
        
        if N == 0:
            keywords: tuple[TfidfKeyword, ...] = tuple()
            vocab_size = 0
        else:
            kept_tokens = [
                tok
                for tok, df in acc.df.items()
                if df >= min_df_abs and df <= max_df_abs
            ]
            
            vocab_size = len(kept_tokens)
            
            scored: list[TfidfKeyword] = []
            for tok in kept_tokens:
                df = acc.df[tok]
                avg_tf = acc.sum_tf_norm[tok] / N
                
                idf = math.log((1.0 + N) / (1.0 + df)) + 1.0
                
                score = avg_tf * idf
                scored.append(
                    TfidfKeyword(
                        token=tok,
                        score=float(score),
                        idf=float(idf),
                        avg_tf=float(avg_tf),
                        df=int(df)
                    )    
                )
                
            scored.sort(key=lambda k: (-k.score, -k.df, k.token)) # desc, desc, asc
            keywords = tuple(scored[: config.top_k])
            
        return TfidfKeywords(
            video_id=video_id,
            silver_path=silver_parquet_path,
            created_at_utc=created_at_utc,
            preprocess_version=self._preprocess_version,
            config_hash=self._hash_config(config),
            row_count=int(acc.row_count),
            empty_text_count=int(acc.empty_text_count),
            doc_count_non_empty=int(acc.doc_count_non_empty),
            vocab_size=int(vocab_size),
            min_df_raw=self._df_cgf_to_str(config.min_df),
            max_df_raw=self._df_cgf_to_str(config.max_df),
            min_df_abs=min_df_abs,
            max_df_abs=max_df_abs,
            config=config,
            keywords=keywords,
        )
    
    @staticmethod
    def _tokenize(text: str, config: TfidfConfig) -> Iterable[str]:
        if config.lowercase:
            text = text.lower()
            
        stopwords = get_stopwords(config.lang) if config.drop_stopwords else frozenset()
        
        for m in _TOKEN_RE.finditer(text): # finditer used since it returns a generator in comparison to findall
            tok = m.group(0) # returns the matched string
            
            if len(tok) < config.min_token_len:
                continue
            if config.drop_numeric_tokens and tok.isdigit():
                continue
            if stopwords and tok in stopwords:
                continue
            
            yield tok
            
    @staticmethod
    def _hash_config(config: TfidfConfig) -> str:
        payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode("utf-8") # dicts are not hashable, so need to convert to json string
        return hashlib.sha256(payload).hexdigest()[:16]
    
    @staticmethod 
    def _df_cgf_to_str(v: int|float) -> str:
        if isinstance(v, bool): # bool is a subclass of int!
            return "1" if v else "0"
        return repr(v) # repr() is used to avoid weird formatting of floats
    

    @staticmethod
    def _resolve_df_thresholds(
        *,
        min_df: int | float,
        max_df: int | float,
        N: int
    ) -> tuple[int, int]:
        """
        Convert min_df/max_df to absolute df thresholds for this corpus size N.

        Rules (frozen):
        - int: absolute document count
        - float in (0,1]: proportion of documents
            min_df_abs = ceil(min_df * N)
            max_df_abs = floor(max_df * N)

        Edge handling:
        - If N == 0: return (1, 0) so nothing can pass.
        - Clamp to [0, N] then ensure min<=max.
        """
        
        if N <= 0:
            return (1, 0)
        
        def to_min_abs(x: int | float) -> int:
            if isinstance(x, int):
                return x
            return int(math.ceil(float(x) * N))
        
        def to_max_abs(x: int | float) -> int:
            if isinstance(x, int):
                return x
            return int(math.floor(float(x) * N))
        
        min_abs = to_min_abs(min_df)
        max_abs = to_max_abs(max_df)
        
        if min_abs < 0:
            min_abs = 0
        if max_abs > N:
            max_abs = N
            
        if max_abs < min_abs:
            pass
        
        return (min_abs, max_abs)
            
        