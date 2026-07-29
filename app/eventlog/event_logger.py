import json
import time
from pathlib import Path


class EventLogger:
    """Writes one JSON row per event to a JSONL file, per Engineering_Document.md §5.5.

    Every row carries the host clock timestamp -- the single source of truth
    for wall-clock time across the whole session (see §5.1).
    """

    def __init__(self, session_dir: Path, session_id: str, participant_id: str):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.participant_id = participant_id
        self._path = self.session_dir / "events.jsonl"
        self._fh = open(self._path, "a", encoding="utf-8")

    def log(
        self,
        event_type: str,
        task: str | None = None,
        block_index: int | None = None,
        trial_index: int | None = None,
        condition_label: str | None = None,
        **payload,
    ) -> None:
        row = {
            "timestamp_host_utc": time.time(),
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "task": task,
            "block_index": block_index,
            "trial_index": trial_index,
            "condition_label": condition_label,
            "event_type": event_type,
            "event_payload_json": payload,
        }
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()
