from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from yt_comments.analysis.distinctive_keywords.models import DistinctiveKeyword, DistinctiveKeywords
from yt_comments.analysis.tfidf.models import TfidfConfig
from yt_comments.storage.gold_distinctive_keywords_repository import ParquetDistinctiveKeywordsRepository



def test_gold_distinctive_keywords_repo_round_trip(tmp_path: Path):
    data_root = tmp_path / "data"
    repo = ParquetDistinctiveKeywordsRepository(data_root=data_root)
    
    config = TfidfConfig()
    item = DistinctiveKeywords(
        channel_id="chan123",
        video_id="v1",
        created_at_utc=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        preprocess_version="v1",
        artifact_version="v1",
        config_hash="deadbeef",
        config=config,
        keyword_count=1,
        keywords=(
            DistinctiveKeyword("cat", 0.63, 0.9, 0.7, 10, 100),
        )
    )
    
    repo.save(item)
    loaded = repo.load("chan123", "v1")
    
    assert item == loaded
    assert isinstance(loaded.keywords, tuple)
    
def test_gold_distinctive_keywords_repo_load_missing_raises(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    repo = ParquetDistinctiveKeywordsRepository(data_root)
    
    with pytest.raises(FileNotFoundError):
        repo.load("missing_chan", "missing_vid")