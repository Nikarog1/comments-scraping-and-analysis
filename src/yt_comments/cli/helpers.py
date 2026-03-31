import argparse
import logging

from datetime import datetime, timezone
from pathlib import Path

from yt_comments.ingestion.scrape_service import ScrapeCommentsService
from yt_comments.ingestion.youtube_api_client import YouTubeApiClient
from yt_comments.storage.bronze_comments_repository import JSONLCommentsRepository
from yt_comments.storage.gold_channel_ref_mapping_repository import JSONChannelRefRepository



logger = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format=LOG_FORMAT)


def _silver_parquet_path(data_root: Path, video_id: str) -> Path:
    return data_root / "silver" / video_id / "comments.parquet"


def _parse_cli_datetime(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime '{value}'. Use ISO format like "
            "2026-01-01 or 2026-01-01T12:00:00"
        ) from e

    # Assuming UTC if tz is missing
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt

def _scrape_video(
          *,
          video_id: str,
          client: YouTubeApiClient,
          repo: JSONLCommentsRepository,
          limit: int | None,
          overwrite: bool,
):
     service = ScrapeCommentsService(client=client, repo=repo)
     return service.run(video_id, overwrite=overwrite, limit=limit)

def _save_channel_id_ref_mapping(*, data_root: str, raw_input: str, channel_id: str) -> Path:
     return JSONChannelRefRepository(data_root=Path(data_root)).save(raw_input=raw_input, channel_id=channel_id)

def _load_channel_id_ref_mapping(*, data_root: str, raw_input: str) -> str:
    try: 
        return JSONChannelRefRepository(data_root=Path(data_root)).load(raw_input=raw_input)
    except Exception as e:
         logger.warning(f"Returning raw input - couldn't find channel's mapping due to: {e}")
         return raw_input
    
def _format_optional_dt(value: datetime | None) -> str:
    if value is None:
        return "N/A"
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")