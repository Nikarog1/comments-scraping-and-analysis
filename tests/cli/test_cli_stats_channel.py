from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from yt_comments.analysis.channel_stats.models import ChannelTokenStat, ChannelTokenStats
from yt_comments.cli.main import main



def test_cli_stats_channel(capsys, tmp_path: Path):
    channel_id = "chan123"
    
    stats = ChannelTokenStats(
        channel_id=channel_id,
        video_ids=("v1", "v2", "v3"),
        created_at_utc=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        preprocess_version="v1",
        config_hash="deadbeef",
        row_count=30,
        empty_text_count=5,
        total_token_count=100,
        unique_token_count=70,
        top_tokens=(
            ChannelTokenStat("cat", 20),
            ChannelTokenStat("dog", 15),
            ChannelTokenStat("amazing", 10)
        )
    )
    
    class FakeSumRepo:
        video_ids = ("v1", "v2", "v3")
    
    mock_summary_repo = Mock()
    mock_summary_repo.load_latest.return_value = FakeSumRepo()
    
    mock_service = Mock()
    mock_service.compute_for_channel.return_value = stats
    
    with (
        patch("yt_comments.cli.main._load_channel_id_ref_mapping", return_value=channel_id),
        patch("yt_comments.cli.main.JSONChannelRunSummaryRepository", return_value=mock_summary_repo),
        patch("yt_comments.cli.main.ChannelTokenStatsService", return_value=mock_service),
    ):
        exit_code = main(
            [
                "stats-channel",
                channel_id,
                "--data-root",
                str(tmp_path),
            ]
        )
        
        assert exit_code == 0
        
        out = capsys.readouterr().out
        
        assert "channel_id: chan123" in out
        assert "videos: 3" in out
        assert "rows: 30" in out
        assert "tokens: total=100 | unique=70" in out
        assert "cat:20" in out