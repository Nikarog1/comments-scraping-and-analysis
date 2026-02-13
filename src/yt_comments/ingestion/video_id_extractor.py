from __future__ import annotations

from urllib.parse import parse_qs, urlparse



def extract_video_id(value: str) -> str:
    """
    Accepts either a raw 11-character YouTube video ID or a YouTube URL
    and return the video ID
    """
    value = value.strip() 
    
    # value is already a video id (not url)
    if "://" not in value:
        return value
    
    parsed = urlparse(value)
    
    # not much needed (except lower()) but kept to avoid NoneType errors
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""
    
    # different examples of domains are tackled here, perhaps too complicated for a pet project, but let's simulate prod :)
    # youtu.be/<id>
    if host.endswith("youtu.be"):
        vid = path.lstrip("/").split("/")[0]
        return vid
    
    # youtube.com/watch?v=<id>
    if path == "/watch":
        qs = parse_qs(parsed.query)
        v = qs.get("v", [None])[0]
        if v:
            return v
    
    # youtube.com/shorts/<id>
    if path.startswith("/shorts/"):
        return path.split("/")[2] # use [2] as path starts with '/'
    
    # youtube.com/embed/<id>
    if path.startswith("/embed/"):
        return path.split("/")[2]
    
    # fallback: return original value if it wasn't covered in the section above (service / API will handle the error)
    return value
    
    


    