from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from yt_comments.analysis.channel_runs.models import ChannelRunSummary
from yt_comments.analysis.channel_stats.models import ChannelTokenStats, TopToken
from yt_comments.analysis.channel_tfidf.models import ChannelTfidfKeywords
from yt_comments.analysis.distinctive_keywords.models import DistinctiveKeyword, DistinctiveKeywords
from yt_comments.analysis.tfidf.models import TfidfConfig, TfidfKeyword, TfidfKeywords
from yt_comments.cli.main import main



def test_cli_report_channel_with_distinctive_keywords_prints_output(capsys, tmp_path: Path):
    data_root = tmp_path / "data"
    channel_id = "chan123"
    video_ids = ("v1", "v2", "v3")
    config = TfidfConfig()
    
    summary = ChannelRunSummary(
        channel_id=channel_id,
        started_at_utc=datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc),
        finished_at_utc=datetime(2026, 2, 1, 12, 1, 0, tzinfo=timezone.utc),
        video_ids=video_ids,
        video_count=len(video_ids),
        comment_count=120,
        error_count=0,
        video_limit=None,
        comment_limit=None,
        published_after=None,
        published_before=None,
    )
    stats = ChannelTokenStats(
        channel_id=channel_id,
        video_ids=video_ids,
        created_at_utc=datetime(2026, 2, 1, 12, 1, 0, tzinfo=timezone.utc),
        preprocess_version="v1",
        config_hash="deadbeef",
        row_count=120,
        empty_text_count=0,
        total_token_count=1200,
        unique_token_count=900,
        top_tokens=(
            TopToken("cat", 60), 
            TopToken("dog", 45),
            TopToken("amazing", 30)
        )
    )
    tfidf = ChannelTfidfKeywords(
        channel_id=channel_id,
        video_ids=video_ids,
        created_at_utc=datetime(2026, 2, 1, 12, 1, 0, tzinfo=timezone.utc),
        preprocess_version="v1",
        artifact_version="v1",
        config_hash="deadbeef",
        row_count=120,
        empty_text_count=0,
        doc_count_non_empty=120,
        vocab_size=1200,
        min_df_raw="2",
        max_df_raw="0.9",
        min_df_abs=2,
        max_df_abs=108,
        config=config,
        keywords=(
            TfidfKeyword("cat", score=0.81, idf=0.9, avg_tf=0.9, df=60),
            TfidfKeyword("dog", score=0.85, idf=0.9, avg_tf=0.91, df=45),
            TfidfKeyword("amazing", score=0.98, idf=0.99, avg_tf=0.99, df=30),
        )
    )    

    distinctive_kws = DistinctiveKeywords(
        channel_id=channel_id,
        video_id=video_ids[0],
        created_at_utc=datetime(2026, 2, 1, 12, 1, 0, tzinfo=timezone.utc),
        preprocess_version="v1",
        artifact_version="v1",
        config_hash="deadbeef",
        config=config,
        keyword_count=3,
        keywords=(
            DistinctiveKeyword(
                "cat", 
                video_score=0.81, 
                channel_score=0.81, 
                lift=1.0, 
                video_df=20, 
                channel_df=60
            ),
            DistinctiveKeyword(
                "dog", 
                video_score=0.64, 
                channel_score=0.85, 
                lift=0.75, 
                video_df=15, 
                channel_df=45
            ),
            DistinctiveKeyword(
                "amazing", 
                video_score=0.49, 
                channel_score=0.98, 
                lift=0.5, 
                video_df=10, 
                channel_df=30
            ),
            )
    )
    
    mock_repo_summary = Mock()
    mock_repo_summary.load_latest.return_value = summary
    
    mock_repo_stats = Mock()
    mock_repo_stats.load.return_value = stats
    
    mock_repo_tfidf = Mock()
    mock_repo_tfidf.load.return_value = tfidf
    
    mock_repo_distinctive_kws = Mock()
    mock_repo_distinctive_kws.load.return_value = distinctive_kws
    
    with (
        patch("yt_comments.cli.commands.channel._load_channel_id_ref_mapping", return_value=channel_id),
        patch("yt_comments.cli.commands.channel.JSONChannelRunSummaryRepository", return_value=mock_repo_summary),
        patch("yt_comments.cli.commands.channel.ParquetChannelTokenStatsRepository", return_value=mock_repo_stats),
        patch("yt_comments.cli.commands.channel.ParquetChannelTfidfKeywordsRepository", return_value=mock_repo_tfidf),
        patch("yt_comments.cli.commands.channel.ParquetDistinctiveKeywordsRepository", return_value=mock_repo_distinctive_kws),
    ):
        exit_code = main(
            [
                "report-channel",
                channel_id,
                "--data-root",
                str(data_root),
                "--video",
                video_ids[0]
            ]
        )
        
    assert exit_code == 0
    
    out = capsys.readouterr().out
    assert f"channel_id: {channel_id}" in out
    assert f"latest_run: videos={summary.video_count}" in out
    assert "run_window: " in out
    assert f"artifacts: preprocess_version={stats.preprocess_version}" in out
    assert "top_tokens:" in out
    assert "cat             count=60" in out
    assert "top_keywords:" in out
    assert "score=0.810" in out
    assert f"distinctive_keywords: video_id={distinctive_kws.video_id}" in out
    assert "lift=1.0" in out
    

def test_cli_report_channel_without_distinctive_keywords_prints_output(capsys, tmp_path: Path):
    data_root = tmp_path / "data"
    channel_id = "chan123"
    video_ids = ("v1", "v2", "v3")
    config = TfidfConfig()
    
    summary = ChannelRunSummary(
        channel_id=channel_id,
        started_at_utc=datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc),
        finished_at_utc=datetime(2026, 2, 1, 12, 1, 0, tzinfo=timezone.utc),
        video_ids=video_ids,
        video_count=len(video_ids),
        comment_count=120,
        error_count=0,
        video_limit=None,
        comment_limit=None,
        published_after=None,
        published_before=None,
    )
    stats = ChannelTokenStats(
        channel_id=channel_id,
        video_ids=video_ids,
        created_at_utc=datetime(2026, 2, 1, 12, 1, 0, tzinfo=timezone.utc),
        preprocess_version="v1",
        config_hash="deadbeef",
        row_count=120,
        empty_text_count=0,
        total_token_count=1200,
        unique_token_count=900,
        top_tokens=(
            TopToken("cat", 60), 
            TopToken("dog", 45),
            TopToken("amazing", 30)
        )
    )
    tfidf = ChannelTfidfKeywords(
        channel_id=channel_id,
        video_ids=video_ids,
        created_at_utc=datetime(2026, 2, 1, 12, 1, 0, tzinfo=timezone.utc),
        preprocess_version="v1",
        artifact_version="v1",
        config_hash="deadbeef",
        row_count=120,
        empty_text_count=0,
        doc_count_non_empty=120,
        vocab_size=1200,
        min_df_raw="2",
        max_df_raw="0.9",
        min_df_abs=2,
        max_df_abs=108,
        config=config,
        keywords=(
            TfidfKeyword("cat", score=0.81, idf=0.9, avg_tf=0.9, df=60),
            TfidfKeyword("dog", score=0.85, idf=0.9, avg_tf=0.91, df=45),
            TfidfKeyword("amazing", score=0.98, idf=0.99, avg_tf=0.99, df=30),
        )
    )    
    
    mock_repo_summary = Mock()
    mock_repo_summary.load_latest.return_value = summary
    
    mock_repo_stats = Mock()
    mock_repo_stats.load.return_value = stats
    
    mock_repo_tfidf = Mock()
    mock_repo_tfidf.load.return_value = tfidf
    
    with (
        patch("yt_comments.cli.commands.channel._load_channel_id_ref_mapping", return_value=channel_id),
        patch("yt_comments.cli.commands.channel.JSONChannelRunSummaryRepository", return_value=mock_repo_summary),
        patch("yt_comments.cli.commands.channel.ParquetChannelTokenStatsRepository", return_value=mock_repo_stats),
        patch("yt_comments.cli.commands.channel.ParquetChannelTfidfKeywordsRepository", return_value=mock_repo_tfidf),
    ):
        exit_code = main(
            [
                "report-channel",
                channel_id,
                "--data-root",
                str(data_root),
            ]
        )
        
    assert exit_code == 0
    
    out = capsys.readouterr().out
    assert f"channel_id: {channel_id}" in out
    assert f"latest_run: videos={summary.video_count}" in out
    assert "run_window: " in out
    assert f"artifacts: preprocess_version={stats.preprocess_version}" in out
    assert "top_tokens:" in out
    assert "cat             count=60" in out
    assert "top_keywords:" in out
    assert "score=0.810" in out
    assert "distinctive_keywords:" not in out
