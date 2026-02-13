from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import requests

from yt_comments.ingestion.models import Comment
from yt_comments.ingestion.youtube_client import YouTubeClient



@dataclass(slots=True)
class YouTubeApiClient(YouTubeClient):
    api_key: str
    
    def fetch_comments(self, video_id: str) -> Iterable[Comment]:
        """
        Fetch top-level comments for a video using YouTube Data API v3
        Note: top-level comments only, pagination supported
        """
        base_url = "https://www.googleapis.com/youtube/v3/commentThreads" # commentThreads returns top-level comments + metadata (2think about comments endpoint which returns replies and ind comments)        
        session = requests.Session()
        
        page_token: Optional[str] = None
        
        while True:
            params = {
                "key": self.api_key,
                "videoId": video_id,
                "part": "snippet",
                "maxResults": 100,
                "textFormat": "plainText",
            }
            if page_token:
                params["pageToken"] = page_token # not relevant for the 1. page, for the rest ensures we send the correct page
                
            resp = session.get(base_url, params=params, timeout=30)
            
            # the below is an error handler 
            try:
                resp.raise_for_status() # for 4xx and 5xx it returns HTTPError
            except requests.HTTPError as e:
                try:
                    # Check if google sends an error message
                    payload = resp.json()
                    err = payload.get("error", {})
                    message = err.get("message") or "Unknown YouTube API error"
                    errors = err.get("errors") or []
                    reason = errors[0].get("reason") if errors else None
                except Exception:
                    # If not or it fails, set vars to None
                    payload = None
                    message = None
                    reason = None
                    
                # Convert different reasons to a readable format
                if reason == "commentsDisabled":
                    raise ValueError(
                        f"Comments are disabled for video '{video_id}'."
                    ) from e

                if reason in {"videoNotFound", "notFound"}:
                    raise ValueError(
                        f"Video '{video_id}' was not found (check the video ID/URL)."
                    ) from e

                if reason in {"quotaExceeded", "dailyLimitExceeded"}:
                    raise ValueError(
                        "YouTube API quota exceeded for this project/API key. "
                        "Try again later or use a different API key/project."
                    ) from e

                if reason in {"keyInvalid", "forbidden"}:
                    raise ValueError(
                        "YouTube API key is invalid or lacks permission for this request."
                    ) from e
                    
                if message:
                    raise ValueError(f"YouTube API error: {message}") from e
                raise
            
            data = resp.json()
            
            items = data.get("items", [])
            for item in items:
                snippet = (
                    item.get("snippet", {})
                    .get("topLevelComment", {})
                    .get("snippet", {})
                ) # youtube returns nested json 
                
                comment_id = item.get("snippet", {}).get("topLevelComment", {}).get("id")
                text = snippet.get("textOriginal") or snippet.get("textDisplay") or ""
                author = snippet.get("authorDisplayName")
                like_count = snippet.get("likeCount")
                published_at_raw = snippet.get("publishedAt")
                
                published_at : Optional[datetime] = None
                if published_at_raw:
                    published_at = datetime.fromisoformat(
                        published_at_raw.replace("Z", "+00:00") # no need to replace Z in python >v3.11 but still keep it if anyone would like to run it on older versions
                    )
                    
                if not comment_id:
                    # defensive approach; if there is a schema drift or unexpected response
                    raise ValueError("YouTube API returned a comment without an id")
                
                yield Comment(
                    video_id=video_id,
                    comment_id=comment_id,
                    text=text,
                    author=author,
                    like_count=like_count,
                    published_at=published_at,
                    is_reply=False,
                ) # generator is created to not store everything in a list at a time. (to avoid RAM leak)
                
            page_token = data.get("nextPageToken")
            if not page_token:
                break
