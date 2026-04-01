from datetime import datetime, timezone
from unittest.mock import Mock

from yt_comments.ingestion.models import Comment
from yt_comments.ingestion.scrape_service import ScrapeCommentsService
from yt_comments.storage.bronze_comments_repository import JSONLCommentsRepository



def test_scrape_service_saves_comments_and_returns_result(tmp_path) -> None:
    video_id = "dQw4w9WgXcQ"
    dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    fake_client = [
            Comment(
                video_id=video_id,
                comment_id="c1",
                text="This is a dummy top-level comment.",
                author="dummy_user_1",
                like_count=3,
                published_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                is_reply=False,
            ),
            Comment(
                video_id=video_id,
                comment_id="c2",
                text="Another dummy top-level comment.",
                author="dummy_user_2",
                like_count=0,
                published_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                is_reply=False,
            ),
            Comment(
                video_id=video_id,
                comment_id="r1",
                text="This is a dummy reply to a comment.",
                author="dummy_user_3",
                like_count=None,
                published_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                is_reply=True,
            ),
        ]
    mock_client = Mock()
    mock_client.fetch_comments.return_value = fake_client
    
    client = mock_client
    repo = JSONLCommentsRepository(tmp_path)
    service = ScrapeCommentsService(client=client, repo=repo)
    
    result = service.run(video_id, overwrite=True)
    
    assert result.video_id == video_id
    assert result.saved_count == 3
    assert result.path.exists()
    
    loaded = repo.load(video_id)
    assert len(loaded) == 3
    assert all(c.video_id == video_id for c in loaded)