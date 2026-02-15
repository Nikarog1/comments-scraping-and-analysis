from __future__ import annotations

import re
import unicodedata



# compiling just once outside of the main class
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE) # if people send url in comms; found during testing some videos
_WS_RE = re.compile(r"\s+")

class TextPreprocessor:
    """
    General text normalization for silver layer
    """
    
    def __init__(self, *, replace_urls_with: str = "<URL>") -> None:
        self._replace_urls_with = replace_urls_with
    
    def clean(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = _URL_RE.sub(self._replace_urls_with, text)
        text = text.lower()
        text = _WS_RE.sub(" ", text).strip()
        
        return text