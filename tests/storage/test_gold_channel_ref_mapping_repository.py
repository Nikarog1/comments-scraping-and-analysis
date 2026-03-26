from pathlib import Path

from yt_comments.storage.gold_channel_ref_mapping_repository import JSONChannelRefRepository


def test_channel_ref_repository_round_trip(tmp_path: Path):
    raw_input = "@test_handle"
    channel_id = "chan123"
    
    repo = JSONChannelRefRepository(data_root=tmp_path)
    path = repo.save(raw_input=raw_input, channel_id=channel_id)
    
    assert path.exists()
    
    got = repo.load(raw_input=raw_input)
    
    assert got == channel_id


