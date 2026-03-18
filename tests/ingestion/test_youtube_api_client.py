from datetime import datetime, timezone 

import pytest
from unittest.mock import Mock, patch

import requests

from yt_comments.ingestion.models import ChannelVideoDiscovery
from yt_comments.ingestion.youtube_api_client import YouTubeApiClient


def test_fetch_comments_single_page():
    client = YouTubeApiClient(api_key="test-key")
    
    response_payload = {
        "items": [
            {
                "snippet": {
                    "topLevelComment": {
                        "id": "comm1",
                        "snippet": {
                            "textOriginal": "Cool vid",
                            "authorDisplayName": "Author1",
                            "likeCount": 4,
                            "publishedAt": "2026-01-01T12:34:56Z", 
                        }
                    }
                }
            },
            {
                "snippet": {
                    "topLevelComment": {
                        "id": "comm2",
                        "snippet": {
                            "textOriginal": "amazing cat",
                            "authorDisplayName": "Author2",
                            "likeCount": 0,
                            "publishedAt": "2026-01-02T09:00:00Z", 
                        }
                    }
                }
            },
        ]
    }
    
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = response_payload
    
    mock_session = Mock()
    mock_session.get.return_value = mock_response
    
    with patch("yt_comments.ingestion.youtube_api_client.requests.Session", return_value=mock_session):
        comments = list(client.fetch_comments(video_id="vid1"))
        
    assert len(comments) == 2
        
    assert comments[0].video_id == "vid1"
    assert comments[0].comment_id == "comm1"
    assert comments[0].text == "Cool vid"
    assert comments[0].author == "Author1"
    assert comments[0].like_count == 4
    assert comments[0].published_at == datetime(2026, 1, 1, 12, 34, 56, tzinfo=timezone.utc)
    
    assert comments[1].video_id == "vid1"
    assert comments[1].comment_id == "comm2"
    assert comments[1].text == "amazing cat"
    assert comments[1].author == "Author2"
    assert comments[1].like_count == 0
    assert comments[1].published_at == datetime(2026, 1, 2, 9, 0, 0, tzinfo=timezone.utc)
    
    mock_session.get.assert_called_once()
    _, kwargs = mock_session.get.call_args
    assert kwargs["timeout"] == 30
    assert kwargs["params"]["key"] == "test-key"
    assert kwargs["params"]["videoId"] == "vid1"
    assert kwargs["params"]["maxResults"] == 100
        
def test_fetch_comments_paginates_across_two_pages():
    client = YouTubeApiClient(api_key="test-key")
    
    page_1 = {
        "items": [
            {
                "snippet": {
                    "topLevelComment": {
                        "id": "comm1",
                        "snippet": {
                            "textOriginal": "Cool vid",
                            "authorDisplayName": "Author1",
                            "likeCount": 4,
                            "publishedAt": "2026-01-01T12:34:56Z", 
                        }
                    }
                }
            },
        ],
        "nextPageToken": "token-2",
    }
    
    page_2 = {
        "items": [
            {
                "snippet": {
                    "topLevelComment": {
                        "id": "comm2",
                        "snippet": {
                            "textOriginal": "amazing cat",
                            "authorDisplayName": "Author2",
                            "likeCount": 0,
                            "publishedAt": "2026-01-02T09:00:00Z", 
                        }
                    }
                }
            },
        ]
    }
    
    response_1 = Mock()
    response_1.raise_for_status.return_value = None
    response_1.json.return_value = page_1
    
    response_2 = Mock()
    response_2.raise_for_status.return_value = None
    response_2.json.return_value = page_2
    
    mock_session = Mock()
    mock_session.get.side_effect = [response_1, response_2]
    
    with patch("yt_comments.ingestion.youtube_api_client.requests.Session", return_value=mock_session):
        comments = list(client.fetch_comments(video_id="vid1"))
        
    assert [comment.comment_id for comment in comments] == ["comm1", "comm2"]

    assert mock_session.get.call_count == 2

    first_call = mock_session.get.call_args_list[0]
    second_call = mock_session.get.call_args_list[1]

    assert "pageToken" not in first_call.kwargs["params"]
    assert second_call.kwargs["params"]["pageToken"] == "token-2"
    

def test_fetch_comments_quota_exceeded_error():
    client = YouTubeApiClient(api_key="test-key")

    error_payload = {
        "error": {
            "message": "Quota exceeded",
            "errors": [
                {"reason": "quotaExceeded"}
            ],
        }
    }

    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError() # didn't work with return_value, side_effect was needed
    response.json.return_value = error_payload

    mock_session = Mock()
    mock_session.get.return_value = response

    with patch(
        "yt_comments.ingestion.youtube_api_client.requests.Session",
        return_value=mock_session,
    ):
        with pytest.raises(ValueError, match="quota exceeded"):
            list(client.fetch_comments(video_id="vid1"))


def test_fetch_comments_disabled_comments_error():
    client = YouTubeApiClient(api_key="test-key")

    error_payload = {
        "error": {
            "message": "Comments disabled",
            "errors": [
                {"reason": "commentsDisabled"}
            ],
        }
    }

    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError()
    response.json.return_value = error_payload

    mock_session = Mock()
    mock_session.get.return_value = response

    with patch(
        "yt_comments.ingestion.youtube_api_client.requests.Session",
        return_value=mock_session,
    ):
        with pytest.raises(ValueError, match="Comments are disabled"):
            list(client.fetch_comments(video_id="vid1"))
            


def test_discover_videos_single_page():
    client = YouTubeApiClient(api_key="test-key")
    
    response_payload = {
        "items": [
            {
                "id": {"videoId": "vid1"},
                "snippet": {
                    "title": "First video",
                    "publishedAt": "2026-01-01T12:34:56Z",
                    "channelId": "chan123",
                },
            },
            {
                "id": {"videoId": "vid2"},
                "snippet": {
                    "title": "Second video",
                    "publishedAt": "2026-01-02T09:00:00Z",
                    "channelId": "chan123",
                },
            },
        ]
    }
    
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = response_payload
    
    mock_session = Mock()
    mock_session.get.return_value = mock_response
    
    request = ChannelVideoDiscovery(
        channel_id="chan123",
        published_after=datetime(2026, 1, 1, tzinfo=timezone.utc),
        limit=10,
        )
    
    with patch("yt_comments.ingestion.youtube_api_client.requests.Session", return_value=mock_session):
        videos = list(client.discover_videos(request=request))
        
    assert len(videos) == 2
    
    assert videos[0].video_id == "vid1"
    assert videos[0].channel_id == "chan123"
    assert videos[0].title == "First video"
    assert videos[0].published_at == datetime(2026, 1, 1, 12, 34, 56, tzinfo=timezone.utc)
    
    assert videos[1].video_id == "vid2"
    assert videos[1].channel_id == "chan123"
    assert videos[1].title == "Second video"
    assert videos[1].published_at == datetime(2026, 1, 2, 9, 0, 0, tzinfo=timezone.utc)
    
    mock_session.get.assert_called_once()
    _, kwargs = mock_session.get.call_args
    assert kwargs["timeout"] == 30
    assert kwargs["params"]["key"] == "test-key"
    assert kwargs["params"]["channelId"] == "chan123"
    assert kwargs["params"]["type"] == "video"
    assert kwargs["params"]["order"] == "date"
    assert kwargs["params"]["maxResults"] == 10
    assert kwargs["params"]["publishedAfter"] == "2026-01-01T00:00:00Z"
    

def test_discover_videos_paginates_across_two_pages():
    client = YouTubeApiClient(api_key="test-key")

    page_1 = {
        "items": [
            {
                "id": {"videoId": "vid1"},
                "snippet": {
                    "title": "First video",
                    "publishedAt": "2026-01-01T12:34:56Z",
                    "channelId": "chan123",
                },
            }
        ],
        "nextPageToken": "token-2",
    }

    page_2 = {
        "items": [
            {
                "id": {"videoId": "vid2"},
                "snippet": {
                    "title": "Second video",
                    "publishedAt": "2026-01-02T09:00:00Z",
                    "channelId": "chan123",
                },
            }
        ]
    }
    
    response_1 = Mock()
    response_1.raise_for_status.return_value = None
    response_1.json.return_value = page_1
    
    response_2 = Mock()
    response_2.raise_for_status.return_value = None
    response_2.json.return_value = page_2
    
    mock_session = Mock()
    mock_session.get.side_effect = [response_1, response_2]
    
    request = ChannelVideoDiscovery(
        channel_id="chan123",
        published_after=datetime(2026, 1, 1, tzinfo=timezone.utc),
        limit=10,
        )
    
    with patch("yt_comments.ingestion.youtube_api_client.requests.Session", return_value=mock_session):
        videos = list(client.discover_videos(request=request))
        
    assert [video.video_id for video in videos] == ["vid1", "vid2"]

    assert mock_session.get.call_count == 2

    first_call = mock_session.get.call_args_list[0]
    second_call = mock_session.get.call_args_list[1]

    assert "pageToken" not in first_call.kwargs["params"]
    assert second_call.kwargs["params"]["pageToken"] == "token-2"
    

def test_discover_videos_respects_video_limit():
    client = YouTubeApiClient(api_key="test-key")
    
    response_payload = {
        "items": [
            {
                "id": {"videoId": "vid1"},
                "snippet": {
                    "title": "First video",
                    "publishedAt": "2026-01-01T12:34:56Z",
                    "channelId": "chan123",
                },
            },
            {
                "id": {"videoId": "vid2"},
                "snippet": {
                    "title": "Second video",
                    "publishedAt": "2026-01-02T09:00:00Z",
                    "channelId": "chan123",
                },
            },
                        {
                "id": {"videoId": "vid3"},
                "snippet": {
                    "title": "Third video",
                    "publishedAt": "2026-01-03T09:00:00Z",
                    "channelId": "chan123",
                },
            },
        ]
    }
    
    mock_response = Mock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = response_payload
    
    mock_session = Mock()
    mock_session.get.return_value = mock_response
    
    request = ChannelVideoDiscovery(
        channel_id="chan123",
        published_after=datetime(2026, 1, 1, tzinfo=timezone.utc),
        limit=2,
        )
    
    with patch("yt_comments.ingestion.youtube_api_client.requests.Session", return_value=mock_session):
        videos = list(client.discover_videos(request=request))
        
    assert [v.video_id for v in videos] == ["vid1", "vid2"]

    assert mock_session.get.call_count == 1
    
def test_discover_videos_quota_exceeded_error():
    client = YouTubeApiClient(api_key="test-key")

    error_payload = {
        "error": {
            "message": "Quota exceeded",
            "errors": [
                {"reason": "quotaExceeded"}
            ],
        }
    }

    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError()
    response.json.return_value = error_payload

    mock_session = Mock()
    mock_session.get.return_value = response

    request = ChannelVideoDiscovery(channel_id="chan123")

    with patch(
        "yt_comments.ingestion.youtube_api_client.requests.Session",
        return_value=mock_session,
    ):
        with pytest.raises(ValueError, match="quota exceeded"):
            list(client.discover_videos(request))
            
def test_discover_videos_passes_published_after_and_before_params():
    client = YouTubeApiClient(api_key="test-key")

    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"items": []}

    mock_session = Mock()
    mock_session.get.return_value = response

    request = ChannelVideoDiscovery(
        channel_id="chan123",
        published_after=datetime(2026, 1, 1, tzinfo=timezone.utc),
        published_before=datetime(2026, 2, 1, tzinfo=timezone.utc),
        limit=10,
    )

    with patch(
        "yt_comments.ingestion.youtube_api_client.requests.Session",
        return_value=mock_session,
    ):
        videos = list(client.discover_videos(request))

    assert videos == []

    mock_session.get.assert_called_once()
    _, kwargs = mock_session.get.call_args

    assert kwargs["params"]["publishedAfter"] == "2026-01-01T00:00:00Z"
    assert kwargs["params"]["publishedBefore"] == "2026-02-01T00:00:00Z"
    
