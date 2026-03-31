from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from yt_comments.cli.main import main
from yt_comments.ingestion.channel_video_discovery_service import ChannelVideoDiscoveryResult
from yt_comments.ingestion.models import ChannelVideo
from yt_comments.ingestion.scrape_service import ScrapeResult




def test_cli_scrape_channel(capsys, tmp_path: Path):

    discovered_videos = [
        ChannelVideo(
            video_id="v1",
            channel_id="UC_test",
            title="Example video 1",
            published_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        ),
        ChannelVideo(
            video_id="v2",
            channel_id="UC_test",
            title="Example video 2",
            published_at=datetime(2026, 1, 2, 10, 45, tzinfo=timezone.utc)
        ),
        ChannelVideo(
            video_id="v3",
            channel_id="UC_test",
            title="Example video 3",
            published_at=datetime(2026, 2, 1, 15, 5, tzinfo=timezone.utc)
        ),
    ]
    discovered_result = ChannelVideoDiscoveryResult(
        video_count=len(discovered_videos),
        videos=discovered_videos,
    )
    
    mock_client = Mock()
    mock_client.resolve_channel_id.return_value = "UC_test"
    
    mock_discovery_service = Mock()
    mock_discovery_service.run.return_value = discovered_result
    
    scrape_result_1 = ScrapeResult(
        video_id="v1", 
        saved_count=10, 
        path=tmp_path / "bronze" / "v1" / "comments.parquet"
    )
    scrape_result_2 = ScrapeResult(
        video_id="v2", 
        saved_count=1, 
        path=tmp_path / "bronze" / "v2" / "comments.parquet"
    )
    scrape_result_3 = ScrapeResult(
        video_id="v3", 
        saved_count=7, 
        path=tmp_path / "bronze" / "v3" / "comments.parquet"
    )
    
    with (
        patch.dict("os.environ", {"YOUTUBE_API_KEY": "test-key"}), 
        patch(
            "yt_comments.cli.commands.channel.YouTubeApiClient",
            return_value=mock_client,
        ), 
        patch(
            "yt_comments.cli.commands.channel.ChannelVideoDiscoveryService",
            return_value=mock_discovery_service,
        ), 
        patch(
            "yt_comments.cli.commands.channel._scrape_video",
            side_effect=[scrape_result_1, scrape_result_2, scrape_result_3],
        ) as mock_scrape_videos
    ):
        exit_code = main(
            [
                "scrape-channel",
                "https://www.youtube.com/@test_handle",
                "--video-limit",
                "3",
                "--bronze-dir",
                str(tmp_path / "bronze"),
                "--data-root",
                str(tmp_path),
            ]
        )
        
    out = capsys.readouterr().out
    
    assert exit_code == 0
    
    assert "v1 | title=Example video 1 | comments=10" in out
    assert "v2 | title=Example video 2 | comments=1" in out
    assert "v3 | title=Example video 3 | comments=7" in out
    assert "TOTAL | videos=3 | comments=18 | errors=0" in out
    
    mock_client.resolve_channel_id.assert_called_once()
    
    assert mock_scrape_videos.call_count == 3