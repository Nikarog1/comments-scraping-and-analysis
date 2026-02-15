from datetime import datetime, timezone

from yt_comments.ingestion.scrape_service import ScrapeResult, ScrapeCommentsService
from yt_comments.ingestion.youtube_client import StubYouTubeClient
from yt_comments.storage.bronze_comments_repository import JSONLCommentsRepository



def test_scrape_service_saves_comments_and_returns_result(tmp_path) -> None:
    video_id = "dQw4w9WgXcQ"
    dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    client = StubYouTubeClient(fixed_now=dt)
    repo = JSONLCommentsRepository(tmp_path)
    service = ScrapeCommentsService(client=client, repo=repo)
    
    result = service.run(video_id, overwrite=True)
    
    assert result.video_id == video_id
    assert result.saved_count == 3
    assert result.path.exists()
    
    loaded = repo.load(video_id)
    assert len(loaded) == 3
    assert all(c.video_id == video_id for c in loaded)