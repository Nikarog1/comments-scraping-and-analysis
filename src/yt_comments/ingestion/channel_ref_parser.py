from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse



_CHANNEL_ID_RE = re.compile(r"^UC[a-zA-Z0-9_-]{22}$")
_HANDLE_RE = re.compile(r"^@[a-z-A-Z0-9._-]{3,30}$")

@dataclass(frozen=True, slots=True)
class ParsedChannelRef:
    kind: str # "channel_id" | "handle" | "username"
    value: str
    
def parse_channel_ref(raw: str) -> ParsedChannelRef:
    value = raw.strip()
    if not value:
        raise ValueError("Channel reference is empty")
    
    # Case 1: direct API channel id
    if _CHANNEL_ID_RE.fullmatch(value):
        return ParsedChannelRef(kind="channel_id", value=value)
    
    # Case 2: handle like @nhl_highlights
    if _HANDLE_RE.fullmatch(value):
        return ParsedChannelRef(kind="handle", value=value)
    
    # Case 3: url forms
    if value.startswith(("http://", "https://")): # not handling cases where url is only www.youtube... without http
        parsed = urlparse(value)
        host = (parsed.netloc or "").lower()
        path = parsed.path or ""
        
        if host in {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "youtube-nocookie.com",
            "www.youtube-nocookie.com",
        }:
            parts = [p for p in path.split("/") if p]
            
            if not parts:
                raise ValueError(f"Unsupported YouTube channel URL: '{raw}'")
            
            # /channel/UC...
            if parts[0] == "channel" and len(parts) >= 2:
                channel_id = parts[1]
                if not _CHANNEL_ID_RE.fullmatch(channel_id):
                    raise ValueError(f"Invalid YouTube channel ID in URL: '{channel_id}'")
                return ParsedChannelRef(kind="channel_id", value=channel_id)
            
            # @handle
            if parts[0].startswith("@"):
                handle = parts[0]
                if not _HANDLE_RE.fullmatch(handle):
                    raise ValueError(f"Invalid YouTube handle in URL: '{handle}'")
                return ParsedChannelRef(kind="handle", value=handle)
            
            # user/name
            if parts[0] == "user" and len(parts) >= 2:
                username = parts[1].strip()
                if not username:
                    raise ValueError(f"Invalid YouTube username URL: '{raw}'")
                return ParsedChannelRef(kind="username", value=username)
            
    raise ValueError(
        "Unsupported channel reference. Use a channel ID (UC...), "
        "a handle (@name), or a YouTube URL in /channel/, /@handle, or /user/ form."
    )
                    
