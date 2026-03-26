from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from yt_comments.ingestion.channel_ref_parser import parse_channel_ref



class JSONChannelRefRepository:
    def __init__(self, data_root: Path):
        self._data_root = data_root
        
    def save(self, raw_input: str, channel_id: str) -> Path:
        normalized_ref = self._normalize_input_ref(raw_input)
        path = self._ref_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        
        payload = self._read_payload(path)
        payload[normalized_ref] = {
            "input_ref": normalized_ref,
            "channel_id": channel_id,
            "resolved_at_utc": (
                datetime.now(tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            ),
        }
        
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            
        return path
    
    
    def load(self, raw_input: str) -> str:
        normalized_ref = self._normalize_input_ref(raw_input)
        path = self._ref_path()
        
        if not path.exists():
            raise FileNotFoundError("Channel refernce mapping doesn't exist")
        
        payload = self._read_payload(path)
        entry = payload.get(normalized_ref)
        if entry is None:
            raise KeyError(f"Channel reference not found in mapping: {raw_input}")
        
        return entry["channel_id"]
        

    def _ref_path(self) -> Path:
        return self._data_root / "gold" / "channel_ref_mapping" / "channel_refs.json"   

    @staticmethod
    def _normalize_input_ref(raw_input: str) -> str:
        parsed = parse_channel_ref(raw=raw_input)
        
        if parsed.kind == "channel_id":
            return parsed.value
        
        if parsed.kind == "handle":
            return parsed.value.lower()
        
        if parsed.kind == "username":
            return parsed.value.lower()
        
        raise ValueError(f"Unsupported channel reference: {raw_input}")
    
    @staticmethod
    def _read_payload(path: Path) -> dict[str, dict]:
        if not path.exists():
            return {}
        
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)