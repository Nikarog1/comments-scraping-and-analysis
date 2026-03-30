from datetime import datetime, timezone

import math
import pytest

from yt_comments.analysis.channel_tfidf.models import ChannelTfidfKeywords
from yt_comments.analysis.distinctive_keywords.service import DistinctiveKeywordsService
from yt_comments.analysis.tfidf.models import TfidfConfig, TfidfKeyword, TfidfKeywords



def test_distinctive_keywords_service_computes_lift_for_video_keywords():
    config = TfidfConfig()

    channel_tfidf = ChannelTfidfKeywords(
        channel_id="chan123",
        video_ids=("v1", "v2", "v3"),
        created_at_utc=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        preprocess_version="v1",
        artifact_version="v1",
        config_hash="deadbeef",
        row_count=105,
        empty_text_count=5,
        doc_count_non_empty=100,
        vocab_size=500,
        min_df_raw="2",
        max_df_raw="0.9",
        min_df_abs=2,
        max_df_abs=90,
        config=config,
        keywords=(
            TfidfKeyword("cat", 0.81, 0.9, 0.9, 10),
            TfidfKeyword("amazing", 0.63, 0.9, 0.7, 8),
            TfidfKeyword("dog", 0.49, 0.7, 0.7, 5),
        )
    )
    
    video_tfidf = TfidfKeywords(
        video_id="v2",
        created_at_utc=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        silver_path="some_path",
        preprocess_version="v1",
        artifact_version="v1",
        config_hash="deadbeef",
        row_count=105,
        empty_text_count=5,
        doc_count_non_empty=100,
        vocab_size=500,
        min_df_raw="2",
        max_df_raw="0.9",
        min_df_abs=2,
        max_df_abs=90,
        config=config,
        keywords=(
            TfidfKeyword("amazing", 0.63, 0.9, 0.7, 8),
            TfidfKeyword("video", 0.49, 0.7, 0.7, 5),
        )
    )
    
    service = DistinctiveKeywordsService()
    result = service.compute_for_video(
        video_tfidf=video_tfidf,
        channel_tfidf=channel_tfidf,
        created_at_utc=datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    
    assert result.channel_id == "chan123"
    assert result.video_id == "v2"
    assert result.keyword_count == 2
    assert result.preprocess_version == "v1"
    assert result.artifact_version == "v1"
    assert result.created_at_utc == datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    got = {kw.token: kw for kw in result.keywords}
    
    assert "amazing" in got
    assert "video" in got
    assert "cat" not in got
    assert got["amazing"].lift == 1.0
    assert got["amazing"].video_df == 8
    assert got["amazing"].channel_df == 8
    assert math.isinf(got["video"].lift)
    assert got["video"].video_df == 5
    assert got["video"].channel_df == 0


def test_distinctive_keywords_service_raises_on_config_hash_mismatch():
    config = TfidfConfig()

    channel_tfidf = ChannelTfidfKeywords(
        channel_id="chan123",
        video_ids=("v1", "v2", "v3"),
        created_at_utc=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        preprocess_version="v1",
        artifact_version="v1",
        config_hash="deadbeef_channel",
        row_count=105,
        empty_text_count=5,
        doc_count_non_empty=100,
        vocab_size=500,
        min_df_raw="2",
        max_df_raw="0.9",
        min_df_abs=2,
        max_df_abs=90,
        config=config,
        keywords=(
            TfidfKeyword("cat", 0.81, 0.9, 0.9, 10),
            TfidfKeyword("amazing", 0.63, 0.9, 0.7, 8),
            TfidfKeyword("dog", 0.49, 0.7, 0.7, 5),
        )
    )
    
    video_tfidf = TfidfKeywords(
        video_id="v2",
        created_at_utc=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        silver_path="some_path",
        preprocess_version="v1",
        artifact_version="v1",
        config_hash="deadbeef_video",
        row_count=105,
        empty_text_count=5,
        doc_count_non_empty=100,
        vocab_size=500,
        min_df_raw="2",
        max_df_raw="0.9",
        min_df_abs=2,
        max_df_abs=90,
        config=config,
        keywords=(
            TfidfKeyword("amazing", 0.63, 0.9, 0.7, 8),
            TfidfKeyword("video", 0.49, 0.7, 0.7, 5),
        )
    )

    service = DistinctiveKeywordsService()

    with pytest.raises(ValueError, match="Config hash mismatch"):
        service.compute_for_video(
            video_tfidf=video_tfidf,
            channel_tfidf=channel_tfidf,
            created_at_utc=datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
    