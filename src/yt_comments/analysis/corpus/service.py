from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from yt_comments.analysis.features import build_document_features, hash_config, read_preprocess_version
from yt_comments.analysis.corpus.contract import CORPUS_ARTIFACT_VERSION
from yt_comments.analysis.corpus.models import CorpusDfTable, CorpusTokenStat
from yt_comments.analysis.tfidf.models import TfidfConfig


class CorpusService:
    """
    Builds corpus-level document frequency statistics across all Silver videos.
    """
    def __init__(self, *, data_root: Path, artifact_version: str = CORPUS_ARTIFACT_VERSION) -> None:
        self._data_root = data_root
        self._artifact_version = artifact_version
        
    def build(self, *, config: TfidfConfig, batch_size: int = 5000) -> CorpusDfTable:
        """
        Build a corpus document-frequency table from all available Silver video datasets.

        Counts each feature at most once per video to produce corpus-level DF values
        suitable for global TF-IDF scoring.

        Args:
            config: Feature extraction settings used to build corpus terms.
            batch_size: Number of rows to read per parquet batch.

        Returns:
            CorpusDfTable artifact with per-feature video frequencies.
        """
        
        silver_root = self._data_root / "silver"
        if not silver_root.exists():
            raise FileNotFoundError("Silver layer not found: no preprocessed comments available.")
        
        feature_video_df: dict[str, int] = defaultdict(int)
        video_count = 0 
        preprocess_version: str | None = None
        
        for video_dir in sorted(silver_root.iterdir()): # iterdir() returns iterators in different orders, sorted helps here to be deterministic 
            silver_parquet_path = video_dir / "comments.parquet" # video_dir is already a FULL path
            
            if not silver_parquet_path.exists():
                continue
            
            current_preprocess_version = read_preprocess_version(silver_parquet_path)
            if preprocess_version is None:
                preprocess_version = current_preprocess_version
            elif current_preprocess_version != preprocess_version:
                raise ValueError("Mixed preprocess_version values found in Silver layer: "
                                 f"expected: {preprocess_version}, got: {current_preprocess_version}"
                                 f"for {silver_parquet_path}"
                )
            
            video_count += 1
            
            features_in_video: set[str] = set() # ensures each token contributes once per video (memory safety)
            
            pf = pq.ParquetFile(silver_parquet_path)
            for batch in pf.iter_batches(batch_size=batch_size, columns=["text_clean"]):
                for text in batch.column(0).to_pylist():
                    if not text:
                        continue
                    
                    doc_features = build_document_features(text, config)
                    features_in_video.update(doc_features) # accepts iterables compared to .add()
                    
            for feature in features_in_video:
                feature_video_df[feature] += 1
                
        tokens_sorted = sorted(
            feature_video_df.items(),
            key=lambda x: (-x[1], x[0]) # df_videos DESC, token ASC
        )
        
        tokens = tuple(
            CorpusTokenStat(token=tok, df_videos=df)
            for tok, df in tokens_sorted
        )
        
        return CorpusDfTable(
            artifact_version=self._artifact_version,
            preprocess_version=preprocess_version,
            config_hash=hash_config(config),
            video_count=video_count,
            tokens=tokens
        )