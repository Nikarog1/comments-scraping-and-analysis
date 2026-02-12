import json
from datetime import datetime, timezone

import pytest

from yt_comments.ingestion.models import Comment
from yt_comments.storage.comments_repository import JSONLCommentsRepository



def test_repo_load_missing_returns_empty_list(tmp_path) -> None:
    """
    Test that loading a missing file returns an empty list
    """
    repo = JSONLCommentsRepository(tmp_path)
    loaded = repo.load("dQw4w9WgXcQ")
    assert loaded == []
    
def test_repo_save_then_load_roundtrip(tmp_path) -> None:
    """
    Test the full cycle and content of jsonl
    """
    repo = JSONLCommentsRepository(tmp_path)
    
    dt = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    
    comments = [
        Comment(
            video_id="dQw4w9WgXcQ",
            comment_id="c1",
            text="hello",
            author="alice",
            like_count=3,
            published_at=dt,
            is_reply=False,
        ),
        Comment(
            video_id="dQw4w9WgXcQ",
            comment_id="c2",
            text="world",
            author=None,
            like_count=None,
            published_at=None,
            is_reply=True,
        ),
    ]
    
    # Check that saving was successful
    path = repo.save("dQw4w9WgXcQ", comments)
    assert path.exists()
    
    # Check that loading is equal to what was saved
    loaded = repo.load("dQw4w9WgXcQ")
    assert loaded == comments
    
    # Check JSONL shape and content
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(comments) # if there is one json object per line
    
    first = json.loads(lines[0])
    assert isinstance(first, dict) # if it's really a json object
    assert first["video_id"] == "dQw4w9WgXcQ" # to be protected against schema drift
    assert isinstance(first["published_at"], str) # if datetime is saved correctly in json
    
def test_repo_load_invalid_record_raises_value_error(tmp_path) -> None:
    """
    If the JSONL file is corrupted, load() fails with ValueError
    """
    repo = JSONLCommentsRepository(tmp_path)
    path = tmp_path / "dQw4w9WgXcQ.jsonl"
    path.write_text('{"ok": 1}\n{"not ok"}\n', encoding="utf-8")
    
    with pytest.raises(ValueError):
        repo.load("dQw4w9WgXcQ")


