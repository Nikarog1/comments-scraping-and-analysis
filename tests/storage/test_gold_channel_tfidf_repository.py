from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from yt_comments.analysis.channel_tfidf.models import ChannelTfidfKeywords
from yt_comments.analysis.tfidf.models import TfidfConfig, TfidfKeyword
from yt_comments.storage.gold_channel_tfidf_repository import ParquetChannelTfidfKeywordsRepository


def test_gold_channel_tfidf_keywords_repo_round_trip(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    repo = ParquetChannelTfidfKeywordsRepository(data_root=data_root)
    
    channel_id = "chan123"
    video_ids = ("v1", "v2", "v3")
    created_at = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    cfg = TfidfConfig(
        top_k=3,
        min_token_len=2,
        drop_numeric_tokens=True,
        lowercase=True,
        drop_stopwords=True,
        stopwords_lang="en",
        min_df=2,
        max_df=0.9,  # example float threshold
        tf_mode="norm",
        idf_mode="smooth_log_plus1_ln",
    )
    
    item = ChannelTfidfKeywords(
            channel_id=channel_id,
            video_ids=video_ids,
            created_at_utc=created_at,
            preprocess_version="v1",
            artifact_version="v2",
            config_hash="testtesttesttest",
            row_count=5,
            empty_text_count=1,
            doc_count_non_empty=4,
            vocab_size=2,
            min_df_raw="2",
            max_df_raw="0.9",
            min_df_abs=2,
            max_df_abs=3,
            config=cfg,
            keywords=(
                TfidfKeyword(token="cat", score=0.42, idf=1.28, avg_tf=0.33, df=2),
                TfidfKeyword(token="amazing", score=0.35, idf=1.28, avg_tf=0.27, df=2),
                TfidfKeyword(token="video", score=0.18, idf=1.69, avg_tf=0.11, df=1),
            ),
    )
    
    repo.save(item)
    loaded = repo.load(channel_id)
    
    assert item == loaded
    assert isinstance(loaded.keywords, tuple)
    
    # check if channel that doesn't exist, raises filenotfounderror
    def test_gold_channel_tfidf_keywords_repo_load_missing_raises(tmp_path: Path) -> None:
        data_root = tmp_path / "data"
        repo = ParquetChannelTfidfKeywordsRepository(data_root)
        
        with pytest.raises(FileNotFoundError):
            repo.load("missing_chan")