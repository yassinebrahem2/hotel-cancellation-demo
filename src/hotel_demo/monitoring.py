"""Monitoring line writer for check runs."""

import datetime
import json
from typing import Any

from hotel_demo.paths import data_dir


def write_monitoring_line(date: str, record: dict[str, Any]) -> dict[str, Any]:
    """Stamp ran_at (current UTC ISO-8601) and append record to monitoring/<date>.jsonl.

    Returns the stamped record.
    """
    ran_at = datetime.datetime.now(datetime.UTC).isoformat()
    stamped: dict[str, Any] = {}
    if "check" in record:
        stamped["check"] = record["check"]
    stamped["ran_at"] = ran_at
    for k, v in record.items():
        if k != "check" and k != "ran_at":
            stamped[k] = v

    mon_dir = data_dir() / "monitoring"
    mon_dir.mkdir(parents=True, exist_ok=True)
    mon_file = mon_dir / f"{date}.jsonl"
    with mon_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(stamped) + "\n")

    return stamped
