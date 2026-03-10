from dataclasses import dataclass


@dataclass(frozen=True) # decided not to use slots=True here as its benifit is relatively small here (memory)
class CorpusTokenStat:
    token: str
    df_videos: int
    
@dataclass(frozen=True) # decided not to use slots=True here as its benifit is relatively small here (memory)
class CorpusDfTable:
    artifact_version: str
    preprocess_version: str | None
    config_hash: str
    video_count: int
    tokens: tuple[CorpusTokenStat, ...]