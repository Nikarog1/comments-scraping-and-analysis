from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from yt_comments.analysis.features import hash_config, tokenize, read_preprocess_version
from yt_comments.analysis.channel.channel_loader import ChannelTextsLoader
from yt_comments.analysis.channel_stats.models import ChannelTokenStat, ChannelTokenStats, ChannelTokenStatsConfig
from yt_comments.storage.silver_comments_repository import ParquetSilverCommentsRepository



class ChannelTokenStatsService:
    def __init__(self) -> None: # no need to have empty __init__ here, python anyways automatically create it
        pass
    
    def compute_for_channel(
            self,
            *,
            channel_id: str,
            video_ids: tuple[str, ...],
            silver_repo: ParquetSilverCommentsRepository,
            config: ChannelTokenStatsConfig,
            created_at_utc: Optional[datetime] = None,
            batch_size: int = 5000
    ) -> ChannelTokenStats:
        if not video_ids:
            raise ValueError("video_ids must not be empty")

        preprocess_version = self._resolve_preprocess_version(
            video_ids=video_ids,
            silver_repo=silver_repo,
        )
        
        created_at_utc = created_at_utc or datetime.now(timezone.utc)
        if created_at_utc.tzinfo is None:
            raise ValueError("created_at_utc must be timezone-aware")
        if created_at_utc.utcoffset() != timezone.utc.utcoffset(created_at_utc):
            raise ValueError("created_at_utc must be in UTC")
        
        loader = ChannelTextsLoader(silver_repo)
        
        row_count = 0
        empty_text_count = 0
        token_counts: Counter[str] = Counter()
        total_token_count = 0

        for comm in loader.iter_texts(video_ids, batch_size=batch_size):
            row_count += 1
            
            if comm is None or str(comm).strip() == "":
                    empty_text_count += 1
                    continue
            
            for tok in tokenize(comm, config):
                token_counts[tok] += 1
                total_token_count += 1
                
        unique_token_count = len(token_counts)
        
        sorted_tokens = sorted(
            token_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )

        top_tokens = tuple(
            ChannelTokenStat(token=token, count=int(count))
            for token, count in sorted_tokens[: config.top_n_tokens]
        )
        
        return ChannelTokenStats(
            channel_id=channel_id,
            video_ids=video_ids,
            created_at_utc=created_at_utc,
            preprocess_version=preprocess_version,
            config_hash=hash_config(config),
            row_count=int(row_count),
            empty_text_count=int(empty_text_count),
            total_token_count=int(total_token_count),
            unique_token_count=int(unique_token_count),
            top_tokens=top_tokens,
        )
    
    @staticmethod
    def _resolve_preprocess_version(
        *,
        video_ids: tuple[str, ...],
        silver_repo: ParquetSilverCommentsRepository,
    ) -> str:
        versions: set[str] = set()

        for video_id in video_ids:
            silver_path = silver_repo._path_for_comments(video_id)
            version = read_preprocess_version(silver_path)
            versions.add(version)

        if not versions:
            raise ValueError("Could not resolve preprocess_version from input videos")

        if len(versions) != 1:
            raise ValueError(
                "Multiple preprocess_version values found across channel videos: "
                f"{sorted(versions)!r}"
            )

        return next(iter(versions))
            
        
