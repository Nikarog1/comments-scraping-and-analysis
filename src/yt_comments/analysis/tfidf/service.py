import math

from datetime import datetime, timezone
from typing import Optional

import pyarrow.parquet as pq

from yt_comments.analysis.features import build_document_features, hash_config, read_preprocess_version
from yt_comments.analysis.tfidf.accumulator import TfidfAccumulator
from yt_comments.analysis.tfidf.models import TfidfConfig, TfidfKeyword, TfidfKeywords


class TfidfService:
    """
    Gold v2 TF-IDF keywords service (per-video corpus).

    Document unit: one comment
    Corpus: all comments of a single video (Silver parquet)
    """
    
    def __init__(self) -> None:
        pass
        
    def compute_for_video(
            self,
            *,
            video_id: str,
            silver_parquet_path: str,
            config: TfidfConfig,
            created_at_utc: Optional[datetime] = None,
            batch_size: int = 5000,
    ) -> TfidfKeywords:
        preprocess_version = read_preprocess_version(silver_parquet_path=silver_parquet_path)
        
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
                features: list[str] = []
                if comm is not None and str(comm).strip() != "":
                    features = build_document_features(str(comm), config)
                acc.add_document(features)
                
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
            kept_tokens = []
            for tok, df in acc.df.items():
                if df < min_df_abs and df > max_df_abs:
                    continue
                if self._ngram_size(tok) >= 2 and df < config.min_ngram_df:
                    continue
                kept_tokens.append(tok)
 
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
            preprocess_version=preprocess_version,
            artifact_version="tfidf_v2_1",
            config_hash=hash_config(config),
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
    
    @staticmethod
    def _ngram_size(feature: str) -> int:
        return feature.count(" ") + 1
            
        