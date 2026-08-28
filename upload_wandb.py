#!/usr/bin/env python3
"""Upload and continuously follow GoRL async JSONL metrics in W&B.

The async training pipeline writes one flushed JSON object per line to
``async_metrics.jsonl``.  This utility first uploads all existing complete
lines, then waits for and uploads newly appended lines until interrupted.

Progress is stored as a byte offset next to the metrics file.  Re-running the
same command therefore resumes both the local reader and the same W&B run
instead of uploading the complete history again.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any


DEFAULT_METRICS_FILE = Path(
    "/root/GoRL/results/rlpd_fm_async_20260822_024244/async_metrics.jsonl"
)
STATE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload existing GoRL async metrics to W&B and keep following the "
            "JSONL file for new metrics."
        )
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=DEFAULT_METRICS_FILE,
        help=f"JSONL metrics file (default: {DEFAULT_METRICS_FILE})",
    )
    parser.add_argument("--project", default="GoRL-robomimic")
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity. By default W&B uses the entity from your login settings.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Stable W&B run ID (default: parent async run directory name).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Displayed W&B run name (default: run ID).",
    )
    parser.add_argument(
        "--group",
        default=None,
        help="W&B group (default: run ID).",
    )
    parser.add_argument(
        "--tags",
        default="gorl,fm,robomimic,async,external-uploader",
        help="Comma-separated W&B tags.",
    )
    parser.add_argument(
        "--mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="W&B mode (default: online).",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Resume-state path (default: .wandb_upload_state.json beside JSONL).",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=1.0,
        help="Delay while waiting for a newly completed line (default: 1.0).",
    )
    parser.add_argument(
        "--no-follow",
        action="store_true",
        help="Upload currently available complete lines and exit.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Ignore and replace local offset state, starting at byte zero. "
            "Use a new --run-id to avoid duplicate W&B history."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/read metrics without importing W&B or changing resume state.",
    )
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    return args


def _default_run_id(metrics_file: Path) -> str:
    parent_name = metrics_file.resolve().parent.name
    return parent_name or "gorl-async-metrics"


def _default_state_file(metrics_file: Path) -> Path:
    return metrics_file.resolve().parent / ".wandb_upload_state.json"


def _file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"device": int(stat.st_dev), "inode": int(stat.st_ino)}


def _read_state(state_file: Path) -> dict[str, Any] | None:
    if not state_file.exists():
        return None
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read resume state {state_file}: {error}") from error
    if state.get("version") != STATE_VERSION:
        raise RuntimeError(
            f"Unsupported state version in {state_file}: {state.get('version')!r}"
        )
    offset = state.get("offset")
    if not isinstance(offset, int) or offset < 0:
        raise RuntimeError(f"Invalid byte offset in {state_file}: {offset!r}")
    return state


def _write_state(
    state_file: Path,
    metrics_file: Path,
    run_id: str,
    offset: int,
    uploaded_events: int,
) -> None:
    identity = _file_identity(metrics_file)
    state = {
        "version": STATE_VERSION,
        "metrics_file": str(metrics_file.resolve()),
        "run_id": run_id,
        "offset": offset,
        "uploaded_events": uploaded_events,
        **identity,
    }
    state_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_file.with_name(f".{state_file.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, state_file)


def _initial_progress(
    metrics_file: Path,
    state_file: Path,
    run_id: str,
    reset: bool,
) -> tuple[int, int]:
    if reset:
        return 0, 0
    state = _read_state(state_file)
    if state is None:
        return 0, 0

    expected_path = str(metrics_file.resolve())
    if state.get("metrics_file") != expected_path:
        raise RuntimeError(
            f"State file belongs to {state.get('metrics_file')}, not {expected_path}. "
            "Choose another --state-file or pass --reset."
        )
    if state.get("run_id") != run_id:
        raise RuntimeError(
            f"State file belongs to W&B run {state.get('run_id')!r}, not {run_id!r}. "
            "Use the recorded --run-id, another --state-file, or --reset."
        )

    identity = _file_identity(metrics_file)
    if any(state.get(key) != value for key, value in identity.items()):
        raise RuntimeError(
            "The metrics file was replaced since the last upload. Use a new "
            "--run-id and --reset after verifying whether its records are new."
        )
    offset = int(state["offset"])
    if metrics_file.stat().st_size < offset:
        raise RuntimeError(
            "The metrics file is shorter than the saved offset (it was likely "
            "truncated). Use a new --run-id and --reset to upload the new file."
        )
    return offset, int(state.get("uploaded_events", 0))


def _prepare_metrics(metrics: dict[str, Any], wandb_module: Any | None) -> dict[str, Any]:
    prepared = dict(metrics)
    video_path = prepared.pop("_video_path", None)
    if video_path is not None:
        video = Path(video_path)
        if not video.exists():
            print(f"WARNING: video does not exist; skipping it: {video}", file=sys.stderr)
        elif wandb_module is not None:
            prepared["video/evaluation"] = wandb_module.Video(str(video))
    return prepared


def _initialize_wandb(args: argparse.Namespace, run_id: str, run_name: str) -> tuple[Any, Any]:
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "wandb is not installed in this Python environment. Run this utility "
            "with the GoRL gorl_robomimic Conda environment."
        ) from error

    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        id=run_id,
        name=run_name,
        group=args.group or run_id,
        tags=tags,
        mode=args.mode,
        resume="allow",
        config={
            "metrics_file": str(args.metrics_file.resolve()),
            "external_metrics_uploader": True,
        },
    )
    # Encoder stages write a cumulative online optimizer step; a stage-local
    # step would make W&B merge/overwrite train points from different versions.
    run.define_metric("pipeline/encoder_step")
    run.define_metric("train/*", step_metric="pipeline/encoder_step")
    run.define_metric("pipeline/decoder_step")
    run.define_metric("decoder/*", step_metric="pipeline/decoder_step")
    return wandb, run


def upload(args: argparse.Namespace) -> int:
    metrics_file = args.metrics_file.expanduser().resolve()
    if not metrics_file.is_file():
        raise FileNotFoundError(f"Metrics file does not exist: {metrics_file}")

    run_id = args.run_id or _default_run_id(metrics_file)
    run_name = args.name or run_id
    state_file = (args.state_file or _default_state_file(metrics_file)).expanduser().resolve()
    offset, uploaded_events = _initial_progress(
        metrics_file, state_file, run_id, args.reset
    )

    wandb_module = None
    run = None
    if not args.dry_run:
        wandb_module, run = _initialize_wandb(args, run_id, run_name)

    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"\nReceived signal {signum}; finishing after the current event...", flush=True)

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)

    session_events = 0
    print(
        f"Metrics: {metrics_file}\n"
        f"W&B run ID: {run_id}\n"
        f"Starting byte offset: {offset}\n"
        f"Previously uploaded events: {uploaded_events}\n"
        f"Mode: {'dry-run' if args.dry_run else args.mode}; "
        f"follow: {not args.no_follow}",
        flush=True,
    )

    try:
        with metrics_file.open("r", encoding="utf-8") as file:
            file.seek(offset)
            while not stop_requested:
                line_start = file.tell()
                line = file.readline()
                if not line:
                    if args.no_follow:
                        break
                    # clear EOF and wait for the writer to append more data
                    file.seek(line_start)
                    time.sleep(args.poll_seconds)
                    continue
                if not line.endswith("\n"):
                    # Never parse or checkpoint a line while another process may
                    # still be writing it.
                    file.seek(line_start)
                    if args.no_follow:
                        print(
                            f"Incomplete final line at byte {line_start}; leaving it for the next run.",
                            flush=True,
                        )
                        break
                    time.sleep(args.poll_seconds)
                    continue

                next_offset = file.tell()
                if not line.strip():
                    if not args.dry_run:
                        _write_state(
                            state_file, metrics_file, run_id, next_offset, uploaded_events
                        )
                    offset = next_offset
                    continue
                try:
                    decoded = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"Invalid complete JSON line beginning at byte {line_start}: {error}"
                    ) from error
                if not isinstance(decoded, dict):
                    raise RuntimeError(
                        f"Expected a JSON object at byte {line_start}, got {type(decoded).__name__}"
                    )

                prepared = _prepare_metrics(decoded, wandb_module)
                if run is not None:
                    run.log(prepared)
                uploaded_events += 1
                session_events += 1
                offset = next_offset
                if not args.dry_run:
                    # Checkpoint only after W&B accepted the log call.
                    _write_state(
                        state_file, metrics_file, run_id, offset, uploaded_events
                    )
                if session_events == 1 or session_events % 100 == 0:
                    print(
                        f"Uploaded/validated {session_events} events this session "
                        f"({uploaded_events} total), offset={offset}",
                        flush=True,
                    )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if run is not None:
            run.finish()

    print(
        f"Stopped cleanly. Processed {session_events} events this session; "
        f"current offset={offset}.",
        flush=True,
    )
    return 0


def main() -> int:
    try:
        return upload(parse_args())
    except (FileNotFoundError, RuntimeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
