"""Small JSONL-based metrics channel for pipeline subprocesses."""

import json
import math
from numbers import Real
from pathlib import Path
import sys
from typing import Any


def _sanitize_json_value(value: Any, path: str, nonfinite_paths: list[str]) -> Any:
    """Return a strict-JSON-compatible copy and record non-finite float paths."""
    if isinstance(value, dict):
        return {
            key: _sanitize_json_value(item, f"{path}.{key}" if path else str(key), nonfinite_paths)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _sanitize_json_value(item, f"{path}[{index}]", nonfinite_paths)
            for index, item in enumerate(value)
        ]
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            nonfinite_paths.append(path or "<root>")
            return None
        return value.item() if hasattr(value, "item") else value
    return value


def append_metrics(metrics_file: str | None, metrics: dict[str, Any]) -> None:
    """Append a strict-JSON metrics event without crashing on NaN or infinity.

    Non-finite floats are represented as JSON ``null`` and diagnosed through
    metadata in the same event. This keeps monitoring failures from terminating
    a long-running training subprocess while preserving which metrics diverged.
    """
    if metrics_file is None:
        return

    nonfinite_paths: list[str] = []
    sanitized = _sanitize_json_value(metrics, "", nonfinite_paths)
    if nonfinite_paths:
        sanitized["metrics/nonfinite_count"] = len(nonfinite_paths)
        sanitized["metrics/nonfinite_keys"] = nonfinite_paths
        print(
            "WARNING: Replaced non-finite metrics with null: "
            + ", ".join(nonfinite_paths),
            file=sys.stderr,
            flush=True,
        )

    path = Path(metrics_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(sanitized, allow_nan=False) + "\n")
        file.flush()