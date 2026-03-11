import math
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from yt_comments.analysis.corpus.models import CorpusDfTable
from yt_comments.analysis.features import build_document_features, hash_config, read_preprocess_version
from yt_comments.analysis.tfidf.accumulator import TfidfAccumulator
from yt_comments.analysis.tfidf.models import TfidfConfig, TfidfKeyword, TfidfKeywords


class TfidfService:
    """
    Build TF-IDF keyword artifacts for one video.

    Default behavior uses the per-video corpus (v2.1).
    If a global corpus artifact is provided, TF is still computed from the
    target video's comments, while IDF is computed from the global corpus.
    """

    def compute_for_video(
        self,
        *,
        video_id: str,
        silver_parquet_path: Path | str,
        config: TfidfConfig,
        global_corpus: CorpusDfTable | None = None,
        created_at_utc: datetime | None = None,
        batch_size: int = 5000,
    ) -> TfidfKeywords:
        
        preprocess_version = read_preprocess_version(silver_parquet_path)
        config_hash = hash_config(config)

        created_at_utc = created_at_utc or datetime.now(timezone.utc)
        if created_at_utc.tzinfo is None:
            raise ValueError("created_at_utc must be timezone-aware")
        if created_at_utc.utcoffset() != timezone.utc.utcoffset(created_at_utc):
            raise ValueError("created_at_utc must be in UTC")

        acc = TfidfAccumulator()

        pf = pq.ParquetFile(silver_parquet_path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=["text_clean"]):
            for comment in batch.column(0).to_pylist():
                features: list[str] = []

                if comment is not None and str(comment).strip() != "":
                    features = build_document_features(str(comment), config)

                acc.add_document(features)

        local_doc_count = acc.doc_count_non_empty

        global_corpus_df: dict[str, int] | None = None
        if global_corpus is None:
            artifact_version = "tfidf_v2_1"
            idf_doc_count = local_doc_count
        else:
            if global_corpus.preprocess_version != preprocess_version:
                raise ValueError(
                    "Global corpus preprocess_version does not match Silver input: "
                    f"{global_corpus.preprocess_version!r} != {preprocess_version!r}"
                )
            if global_corpus.config_hash != config_hash:
                raise ValueError(
                    "Global corpus config_hash does not match TF-IDF config: "
                    f"{global_corpus.config_hash!r} != {config_hash!r}"
                )

            global_corpus_df = {row.token: row.df_videos for row in global_corpus.tokens}
            artifact_version = "tfidf_v3"
            idf_doc_count = global_corpus.video_count

        min_df_abs, max_df_abs = self._resolve_df_thresholds(
            min_df=config.min_df,
            max_df=config.max_df,
            N=idf_doc_count,
        )

        if local_doc_count == 0:
            keywords: tuple[TfidfKeyword, ...] = tuple()
            vocab_size = 0
        else:
            kept_tokens: list[tuple[str, int]] = []

            for tok in acc.df:
                if global_corpus_df is None:
                    df = acc.df[tok]
                else:
                    if tok not in global_corpus_df:
                        df = 0
                    else:
                        df = global_corpus_df[tok]

                if df < min_df_abs or df > max_df_abs:
                    continue

                if self._ngram_size(tok) >= 2 and df < config.min_ngram_df:
                    continue

                kept_tokens.append((tok, df))

            vocab_size = len(kept_tokens)

            scored: list[TfidfKeyword] = []
            for tok, df in kept_tokens:
                avg_tf = acc.sum_tf_norm[tok] / local_doc_count # global corpus goes to idf only, tf is always about current document, i.e. N here is the number of comms in one video for TFIDF v3
                idf = math.log((1.0 + idf_doc_count) / (1.0 + df)) + 1.0
                score = avg_tf * idf

                scored.append(
                    TfidfKeyword(
                        token=tok,
                        score=float(score),
                        idf=float(idf),
                        avg_tf=float(avg_tf),
                        df=int(df),
                    )
                )

            scored.sort(key=lambda k: (-k.score, -k.df, k.token))
            keywords = tuple(scored[: config.top_k])

        return TfidfKeywords(
            video_id=video_id,
            silver_path=str(silver_parquet_path),
            created_at_utc=created_at_utc,
            preprocess_version=preprocess_version,
            artifact_version=artifact_version,
            config_hash=config_hash,
            row_count=int(acc.row_count),
            empty_text_count=int(acc.empty_text_count),
            doc_count_non_empty=int(acc.doc_count_non_empty),
            vocab_size=int(vocab_size),
            min_df_raw=self._df_cfg_to_str(config.min_df),
            max_df_raw=self._df_cfg_to_str(config.max_df),
            min_df_abs=min_df_abs,
            max_df_abs=max_df_abs,
            config=config,
            keywords=keywords,
        )

    @staticmethod
    def _df_cfg_to_str(v: int | float) -> str:
        """
        Convert a DF config value to a stable string representation.
        """
        if isinstance(v, bool):
            return "1" if v else "0"
        return repr(v)

    @staticmethod
    def _resolve_df_thresholds(
        *,
        min_df: int | float,
        max_df: int | float,
        N: int,
    ) -> tuple[int, int]:
        """
        Convert min_df/max_df config values to absolute document-frequency thresholds.
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
        """
        Return the number of tokens in a feature string.
        """
        return feature.count(" ") + 1
            
        