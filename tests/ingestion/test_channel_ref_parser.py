import pytest

from yt_comments.ingestion.channel_ref_parser import ParsedChannelRef, parse_channel_ref


def test_parse_channel_ref_accepts_direct_channel_id():
    channel_id = parse_channel_ref("UC_123456789ABCDEFH-5632")
    
    assert channel_id == ParsedChannelRef(kind="channel_id", value="UC_123456789ABCDEFH-5632")
    
def test_parse_channel_ref_accepts_handle():
    channel_id = parse_channel_ref("@Test_channel")
    
    assert channel_id == ParsedChannelRef(kind="handle", value="@Test_channel")
    
def test_parse_channel_ref_accepts_url_with_channel_id():
    channel_id = parse_channel_ref("https://www.youtube.com/channel/UC_123456789ABCDEFH-5632")
    
    assert channel_id == ParsedChannelRef(kind="channel_id", value="UC_123456789ABCDEFH-5632")
    
def test_parse_channel_ref_accepts_url_with_handle():
    channel_id = parse_channel_ref("https://www.youtube.com/@Test_channel")
    
    assert channel_id == ParsedChannelRef(kind="handle", value="@Test_channel")
    
def test_parse_channel_ref_accepts_url_with_username():
    channel_id = parse_channel_ref("https://www.youtube.com/user/test_user")
    
    assert channel_id == ParsedChannelRef(kind="username", value="test_user")
    
def test_parse_channel_ref_unsupported_channel_ref_returns_error():
    with pytest.raises(ValueError, match="Unsupported channel reference"):
        parse_channel_ref("show_me_error")