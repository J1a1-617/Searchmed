from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union


_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|authorization|password|passwd|secret|access[_-]?token|refresh[_-]?token)$",
    re.IGNORECASE,
)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _SENSITIVE_KEY.search(str(key)) else _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(model_dump(mode="json"))
        except TypeError:
            return _jsonable(model_dump())
    return str(value)


class TrajectoryEventStore:
    """Append-only, crash-tolerant JSONL event log for one Agent attempt."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: Union[Path, str],
        *,
        context: Optional[Dict[str, Any]] = None,
        fsync_each_event: Optional[bool] = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.context = _jsonable(context or {})
        self.fsync_each_event = (
            bool(fsync_each_event)
            if fsync_each_event is not None
            else os.environ.get("TRAJECTORY_FSYNC", "0") == "1"
        )
        self._lock = threading.Lock()
        self._seq = self._recover_last_seq()
        # A process may die in the middle of one JSON write. Preserve that
        # forensic fragment as an invalid line, but ensure later valid events
        # start on a fresh line and remain replayable.
        if self.path.is_file() and self.path.stat().st_size:
            with self.path.open("rb+") as repair_handle:
                repair_handle.seek(-1, os.SEEK_END)
                if repair_handle.read(1) != b"\n":
                    repair_handle.seek(0, os.SEEK_END)
                    repair_handle.write(b"\n")
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)

    def _recover_last_seq(self) -> int:
        if not self.path.is_file():
            return 0
        last_seq = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    last_seq = max(last_seq, int(event.get("seq") or 0))
        return last_seq

    def append(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not str(event_type).strip():
            raise ValueError("event_type cannot be empty")
        with self._lock:
            self._seq += 1
            event = {
                "schema_version": self.SCHEMA_VERSION,
                "seq": self._seq,
                "event_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **self.context,
                "type": str(event_type),
                "data": _jsonable(data or {}),
            }
            self._handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
            self._handle.flush()
            if self.fsync_each_event:
                os.fsync(self._handle.fileno())
            return event

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.flush()
                self._handle.close()

    def __enter__(self) -> "TrajectoryEventStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
