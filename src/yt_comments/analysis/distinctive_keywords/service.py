from datetime import datetime, timezone

from yt_comments.analysis.channel_tfidf.models import ChannelTfidfKeywords
from yt_comments.analysis.distinctive_keywords.models import DistinctiveKeyword, DistinctiveKeywords
from yt_comments.analysis.tfidf.models import TfidfKeywords


class DistinctiveKeywordsService:
    def compute_for_video(
        self,
        *,
        video_tfidf: TfidfKeywords,
        channel_tfidf: ChannelTfidfKeywords,
        created_at_utc: datetime | None = None,
    ) -> DistinctiveKeywords:
        if video_tfidf.artifact_version != channel_tfidf.artifact_version:
            raise ValueError(
                "Artifact version mismatch between video and channel TF-IDF: "
                f"{video_tfidf.artifact_version} vs {channel_tfidf.artifact_version}"
            )

        if video_tfidf.preprocess_version != channel_tfidf.preprocess_version:
            raise ValueError(
                "Preprocess version mismatch between video and channel TF-IDF: "
                f"{video_tfidf.preprocess_version} vs {channel_tfidf.preprocess_version}"
            )

        if video_tfidf.config_hash != channel_tfidf.config_hash:
            raise ValueError(
                "Config hash mismatch between video and channel TF-IDF: "
                f"{video_tfidf.config_hash} vs {channel_tfidf.config_hash}"
            )

        if video_tfidf.video_id not in channel_tfidf.video_ids:
            raise ValueError(
                f"Provided video_id not found in channel video_ids: {video_tfidf.video_id}"
            )

        created_at_utc = created_at_utc or datetime.now(timezone.utc)
        if created_at_utc.tzinfo is None:
            raise ValueError("created_at_utc must be timezone-aware")
        if created_at_utc.utcoffset() != timezone.utc.utcoffset(created_at_utc):
            raise ValueError("created_at_utc must be in UTC")

        channel_tokens = {
            kw.token: (float(kw.score), int(kw.df))
            for kw in channel_tfidf.keywords
        }

        result: list[DistinctiveKeyword] = []

        for video_kw in video_tfidf.keywords:
            channel_score, channel_df = channel_tokens.get(video_kw.token, (0.0, 0))

            if channel_score == 0:
                lift = float("inf")
            else:
                lift = float(video_kw.score) / float(channel_score)

            result.append(
                DistinctiveKeyword(
                    token=video_kw.token,
                    video_score=float(video_kw.score),
                    channel_score=float(channel_score),
                    lift=float(lift),
                    video_df=int(video_kw.df),
                    channel_df=int(channel_df),
                )
            )

        result.sort(key=lambda k: (-k.lift, -k.video_score, k.token))
        keywords = tuple(result[: video_tfidf.config.top_k])

        return DistinctiveKeywords(
            channel_id=channel_tfidf.channel_id,
            video_id=video_tfidf.video_id,
            created_at_utc=created_at_utc,
            preprocess_version=channel_tfidf.preprocess_version,
            artifact_version=channel_tfidf.artifact_version,
            config_hash=video_tfidf.config_hash,
            config=video_tfidf.config,
            keyword_count=len(keywords),
            keywords=keywords,
        )
                