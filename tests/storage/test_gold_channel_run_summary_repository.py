from datetime import datetime, timezone
from pathlib import Path

from yt_comments.analysis.channel_runs.models import ChannelRunSummary
from yt_comments.storage.gold_channel_run_summary_repository import JSONChannelRunSummaryRepository


def test_channel_run_summary_repository_save_creates_timestamped_json(tmp_path: Path):
    
    summary = ChannelRunSummary(
        channel_id="chan123",
        started_at_utc=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        finished_at_utc=datetime(2026, 1, 1, 12, 1, 30, tzinfo=timezone.utc),
        video_ids=("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"),
        video_count=10,
        comment_count=100,
        error_count=5,
        video_limit=10,
        comment_limit=100,
        published_after=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        published_before=None,
    )
    repo = JSONChannelRunSummaryRepository(data_root=tmp_path)
    path = repo.save(summary)
    
    assert path.exists()
    assert path.name == "20260101T120130Z.json"
    assert path.parent == tmp_path / "gold" / "channel_runs" / "chan123"
    
    text = path.read_text(encoding="utf-8")
    assert '"channel_id": "chan123"' in text
    assert '"video_count": 10' in text
    assert '"comment_count": 100' in text
    assert '"error_count": 5' in text
    assert '"started_at_utc": "2026-01-01T12:00:00Z"' in text
    assert '"finished_at_utc": "2026-01-01T12:01:30Z"' in text
    assert '"published_after": "2026-01-01T00:00:00Z"' in text
    assert '"published_before": null' in text