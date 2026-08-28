#!/usr/bin/env python3
"""Rewrite legacy async train metrics to use cumulative encoder steps.

Older async runs reset ``pipeline/encoder_step`` to zero for every Encoder_n
stage. W&B custom-step charts therefore merged train points from different
versions. This utility preserves the source JSONL and writes a repaired copy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def repair_encoder_steps(source: Path, output: Path) -> tuple[int, int]:
    """Write repaired metrics and return (event_count, repaired_train_events)."""
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if source == output:
        raise ValueError("Output must differ from the source metrics file.")
    if not source.is_file():
        raise FileNotFoundError(f"Metrics file does not exist: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    version_offsets: dict[int, int] = {}
    previous_version: int | None = None
    previous_version_max_step = 0
    cumulative_offset = 0
    event_count = 0
    repaired_train_events = 0

    try:
        with source.open("r", encoding="utf-8") as input_file, temporary.open(
            "w", encoding="utf-8"
        ) as output_file:
            for line_number, line in enumerate(input_file, 1):
                if not line.endswith("\n"):
                    # The source may still be actively appended. Never copy a
                    # partial final event.
                    break
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError(
                        f"Expected JSON object on line {line_number}, got "
                        f"{type(event).__name__}."
                    )

                has_train_metric = any(key.startswith("train/") for key in event)
                if has_train_metric and "pipeline/encoder_step" in event:
                    version = int(event["pipeline/version"])
                    local_step = int(event["pipeline/encoder_step"])
                    if version not in version_offsets:
                        if previous_version is not None:
                            cumulative_offset += previous_version_max_step
                        version_offsets[version] = cumulative_offset
                        previous_version = version
                        previous_version_max_step = 0
                    elif version != previous_version:
                        raise ValueError(
                            "Train metric versions are not contiguous in source order: "
                            f"line {line_number} returned to version {version}."
                        )
                    previous_version_max_step = max(previous_version_max_step, local_step)
                    event["pipeline/encoder_step"] = version_offsets[version] + local_step
                    repaired_train_events += 1

                output_file.write(json.dumps(event, allow_nan=False) + "\n")
                event_count += 1
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return event_count, repaired_train_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_file", type=Path, help="Legacy async_metrics.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: async_metrics_repaired.jsonl beside the source)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.metrics_file.expanduser().resolve()
    output = args.output or source.with_name("async_metrics_repaired.jsonl")
    events, repaired = repair_encoder_steps(source, output)
    print(f"Wrote {events} events to {output}; repaired {repaired} train events.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())