from typing import Iterator

from yt_comments.storage.silver_comments_repository import ParquetSilverCommentsRepository


class ChannelTextsLoader:
    def __init__(self, silver_repo: ParquetSilverCommentsRepository):
        self._silver_repo = silver_repo

    def iter_texts(self, video_ids: tuple[str, ...]) -> Iterator[str]:
        for video_id in video_ids:
            table = self._silver_repo.load(video_id=video_id)
            for text in table["text_clean"].to_pylist():
                if text:
                    yield text