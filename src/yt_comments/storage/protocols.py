from __future__ import annotations 

from typing import Iterable, Protocol

from yt_comments.ingestion.models import Comment



# Adding this abstraction layer to simulate prod; here it'd help us in case of switching to S3 or db in bronze layer
class BronzeCommentsReader(Protocol):  
    def load(self, video_id: str) -> Iterable[Comment]:
        ...
        
class SilverCommentsWriter(Protocol):
    def save(self, video_id: str, df, *, overwrite: bool = True):
        ...
    