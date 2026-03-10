from typing import Protocol

from yt_comments.analysis.corpus.models import CorpusDfTable


class CorpusDfRepository(Protocol):
    """Persistence contract for corpus document-frequency artifacts."""

    def save(self, table: CorpusDfTable) -> None:
        ...

    def load(self) -> CorpusDfTable:
        ...