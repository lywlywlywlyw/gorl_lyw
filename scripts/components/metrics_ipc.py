"""Small JSONL-based metrics channel for pipeline subprocesses."""

import json
from pathlib import Path
from typing import Any


def append_metrics(metrics_file: str | None, metrics: dict[str, Any]) -> None:
    """Atomically append one JSON-serializable metrics event to a JSONL file."""
    if metrics_file is None:
        return

    path = Path(metrics_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metrics, allow_nan=False) + "\n")
        file.flush()