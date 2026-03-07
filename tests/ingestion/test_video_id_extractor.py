from yt_comments.ingestion.video_id_extractor import extract_video_id



def test_extract_video_id_raw_id() -> None:
    assert extract_video_id("id_1") == "id_1"
    
def test_extract_video_id_youtu_be_url() -> None:
    assert extract_video_id("https://youtu.be/id_1") == "id_1"
    
def test_extract_video_id_youtube_watch_url() -> None:
    assert extract_video_id("https://youtube.com/watch?v=id_1") == "id_1"
    
def test_extract_video_id_youtube_shorts_url() -> None:
    assert extract_video_id("https://youtube.com/shorts/id_1") == "id_1"
    
def test_extract_video_id_youtube_embed_url() -> None:
    assert extract_video_id("https://youtube.com/embed/id_1") == "id_1"
    
def test_extract_video_id_raw_id_strimmed() -> None:
    assert extract_video_id("   id_1   ") == "id_1"
    