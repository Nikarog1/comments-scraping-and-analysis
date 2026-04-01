from typing import Iterator
import pyarrow.parquet as pq

from yt_comments.storage.silver_comments_repository import ParquetSilverCommentsRepository


class ChannelTextsLoader:
    """Streams preprocessed comment texts for a set of videos from the Silver layer."""
    def __init__(self, silver_repo: ParquetSilverCommentsRepository):
        self._silver_repo = silver_repo

    def iter_texts(self, video_ids: tuple[str, ...], batch_size: int = 5000) -> Iterator[str]:
        """
        Lazily iterate over cleaned comment texts for the given videos.

        Reads parquet files in batches to avoid loading full datasets into memory.

        Args:
            video_ids: Ordered collection of video identifiers.
            batch_size: Number of rows to read per batch.

        Yields:
            Cleaned comment text strings.
        """
        for video_id in video_ids:
            path = self._silver_repo._path_for_comments(video_id=video_id)
            
            table = pq.ParquetFile(path)
            for batch in table.iter_batches(batch_size=batch_size, columns=["text_clean"]):
                for text in batch.column(0).to_pylist():
                    yield text