from datetime import datetime, timezone
from unittest.mock import Mock, patch

from yt_comments.cli.main import main
from yt_comments.ingestion.channel_video_discovery_service import ChannelVideoDiscoveryResult
from yt_comments.ingestion.models import ChannelVideo


def test_cli_discover_videos(capsys):
    
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
    ):
        
        exit_code = main(
            [
                "discover-videos",
                "@chan123",
                "--limit",
                "3",
                "--published-after",
                "2026-01-01",
            ]
        )

    assert exit_code == 0
    
    out = capsys.readouterr().out
    lines = out.strip().split("\n")
    
    actual = []
    for line in lines[:-1]:
        date_str, video_id, title = [part.strip() for part in line.split("|")]
        actual.append((date_str, video_id, title))

    expected = [
        ("2026-01-01 12:00", "v1", "Example video 1"),
        ("2026-01-02 10:45", "v2", "Example video 2"),
        ("2026-02-01 15:05", "v3", "Example video 3"),
    ]
    
    assert actual == expected

