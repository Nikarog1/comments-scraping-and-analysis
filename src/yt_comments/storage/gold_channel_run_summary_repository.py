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
    def __init__(self, data_root: Path):
        self.data_root = data_root
        
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
    
    def _dir_for_channel(self, channel_id: str) -> Path:
        return self.data_root / "gold" / "channel_runs" / channel_id
    
    def _path_for_summary(self, channel_id: str, json_name: str) -> Path:
        return self._dir_for_channel(channel_id) / f"{json_name}.json"
    