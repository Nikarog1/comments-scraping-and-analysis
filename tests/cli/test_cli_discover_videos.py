from unittest.mock import patch

from yt_comments.cli.main import main
from yt_comments.ingestion.channel_video_discovery_client import StubChannelVideoDiscoveryClient


def test_cli_discover_videos(capsys):
    
    with (
        patch.dict("os.environ", {"YOUTUBE_API_KEY": "test-key"}),
        patch(
            "yt_comments.cli.main.YouTubeApiClient",
            return_value=StubChannelVideoDiscoveryClient()
        )
    ):
        
        exit_code = main(
            [
                "discover_videos",
                "chan123",
                "--video-limit",
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

