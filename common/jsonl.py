"""Per-row JSONL output with resume.

Long jobs write one JSON line per finished row. On restart, rows already in the
file are skipped, so a crash or a stopped machine costs at most one row.
"""

import json
import os
from typing import Iterable, Iterator


def done_keys(path: str, key: str = "id") -> set:
    """Keys of rows already written. A torn last line (crash mid-write) is ignored."""
    keys = set()
    if not os.path.exists(path):
        return keys
    with open(path) as f:
        for line in f:
            try:
                keys.add(json.loads(line)[key])
            except (json.JSONDecodeError, KeyError):
                continue
    return keys


def pending(rows: Iterable[dict], path: str, key: str = "id") -> Iterator[dict]:
    """Yield only the rows whose key is not yet in the output file."""
    seen = done_keys(path, key)
    for row in rows:
        if row[key] not in seen:
            yield row


def append(path: str, record: dict) -> None:
    """Append one record and flush it to disk before returning."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())
