from datetime import datetime, timezone

from yt_comments.ingestion.channel_video_discovery_client import StubChannelVideoDiscoveryClient
from yt_comments.ingestion.channel_video_discovery_service import ChannelVideoDiscoveryService, ChannelVideoDiscoveryResult
from yt_comments.ingestion.models import ChannelVideoDiscovery, ChannelVideo


def test_channel_video_discovery_service_returns_result():
    channel_id = "chan123"
    client = StubChannelVideoDiscoveryClient()
    request = ChannelVideoDiscovery(channel_id=channel_id)
    
    service = ChannelVideoDiscoveryService(client=client, request=request)
    result = service.run()
    
    expected = [ 
        ChannelVideo(
            video_id="v1",
            channel_id=request.channel_id,
            title="Example video 1",
            published_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        ),
        ChannelVideo(
            video_id="v2",
            channel_id=request.channel_id,
            title="Example video 2",
            published_at=datetime(2026, 1, 2, 10, 45, tzinfo=timezone.utc)
        ),
        ChannelVideo(
            video_id="v3",
            channel_id=request.channel_id,
            title="Example video 3",
            published_at=datetime(2026, 2, 1, 15, 5, tzinfo=timezone.utc)
        ),
    ]
    
    assert result.video_count == 3
    assert result.videos == expected
    