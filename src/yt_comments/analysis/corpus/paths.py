from __future__ import annotations

from pathlib import Path


def tfidf_corpus_df_path(root: Path) -> Path:
    """
    Deterministic layout:
      data/gold/corpus_df/corpus.parquet
    """
    return root / "gold" / "corpus_df" / "corpus.parquet"   