from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, date, timezone
from pathlib import Path

from yt_comments.analysis.channel_runs.models import ChannelRunSummary


class JSONChannelRunSummaryRepository:
    """
    JSON repository to store metadata of channel runs (cli command scrape-channel)
    
    Layout:
      data/gold/channel_runs/<channel_id>/stats.parquet
    """
    def __init__(self, data_root: Path | str = "data"):
        self.data_root = Path(data_root)
        
    def save(self,  summary: ChannelRunSummary) -> Path:
        
        run_ts = (
            summary.finished_at_utc
            .astimezone(tz=timezone.utc)
            .strftime("%Y%m%dT%H%M%SZ") # to store files based on finished time
        )
        out_dir = self._dir_for_channel(summary.channel_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        path = self._path_for_summary(summary.channel_id, run_ts)
            
        payload = asdict(summary)
        for key, value in payload.items():
            if isinstance(value, (datetime, date)): # json.dump cannot serialize datetime objects
                payload[key] = value.isoformat().replace("+00:00", "Z")

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)

        return path
    
    def load_latest(self, channel_id: str) -> ChannelRunSummary:
        path = self._latest_summary_path(channel_id)
        
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            
        return ChannelRunSummary(
            channel_id=data["channel_id"],
            started_at_utc=datetime.fromisoformat(data["started_at_utc"]),
            finished_at_utc=datetime.fromisoformat(data["finished_at_utc"]),
            video_ids=tuple(data["video_ids"]), 
            video_count=data["video_count"],
            comment_count=data["comment_count"],
            error_count=data["error_count"],
            video_limit=data["video_limit"],
            comment_limit=data["comment_limit"],
            published_after=(
                datetime.fromisoformat(data["published_after"])
                if data["published_after"] else None
            ),
            published_before=(
                datetime.fromisoformat(data["published_before"])
                if data["published_before"] else None
            ),
        )
    
    def _dir_for_channel(self, channel_id: str) -> Path:
        return self.data_root / "gold" / "channel_runs" / channel_id
    
    def _path_for_summary(self, channel_id: str, json_name: str) -> Path:
        return self._dir_for_channel(channel_id) / f"{json_name}.json"

    def _latest_summary_path(self, channel_id: str) -> Path:
        out_dir = self._dir_for_channel(channel_id=channel_id) 
        
        if not out_dir.exists():
            raise FileNotFoundError(f"No metadata directory for channel_id={channel_id}")
        
        files = list(out_dir.glob("*.json")) # lists all available json within directory as Path("file_name.json")
        if not files:
            raise FileNotFoundError(f"No metadata files for channel_id={channel_id}")
        
        return max(files, key=lambda p: p.stem) # stem removes file extension, i.e., ".json" in that case
    