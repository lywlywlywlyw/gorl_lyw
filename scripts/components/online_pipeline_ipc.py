"""Filesystem IPC primitives for asynchronous Offline -> Online training.

The protocol is deliberately append-only:

* collectors atomically append immutable replay chunks;
* trainers read a stable chunk-list snapshot at stage start;
* checkpoints are written to temporary directories and published by rename;
* a policy is visible only after both matching checkpoints and ``READY`` exist.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TRANSITION_KEYS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "masks",
    "dones",
    "truncations",
    "env_states",
)


def atomic_pickle_dump(value: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def atomic_json_dump(value: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, target)
    return target


def load_transition_data(path: str | Path) -> dict[str, np.ndarray]:
    """Load legacy or canonical transition dictionaries into one schema."""
    with Path(path).expanduser().open("rb") as file:
        raw = pickle.load(file)
    if not isinstance(raw, dict):
        raise TypeError(f"Transition buffer must be a dictionary: {path}")

    aliases = {
        "observations": ("observations", "states", "obs"),
        "actions": ("actions", "env_actions"),
        "rewards": ("rewards", "reward"),
        "next_observations": ("next_observations", "next_states", "next_obs"),
        "masks": ("masks", "discounts", "discount"),
        "dones": ("dones", "terminals", "done"),
        "truncations": ("truncations", "timeouts", "truncated"),
        "env_states": ("env_states", "sim_states"),
    }
    result: dict[str, np.ndarray] = {}
    for target, candidates in aliases.items():
        value = next((raw[name] for name in candidates if name in raw), None)
        if value is not None:
            result[target] = np.asarray(value)

    if "observations" not in result or "actions" not in result:
        raise KeyError(f"Transition buffer {path} must contain states and actions.")
    size = len(result["observations"])
    if "rewards" not in result:
        result["rewards"] = np.zeros(size, dtype=np.float32)
    if "next_observations" not in result:
        # Demonstration files occasionally contain only (s, a). This fallback is
        # sufficient for inverse-FM construction but callers should prefer full
        # transitions when training reward-based critics.
        result["next_observations"] = result["observations"].copy()
    if "dones" not in result:
        result["dones"] = np.zeros(size, dtype=np.bool_)
    if "truncations" not in result:
        result["truncations"] = np.zeros(size, dtype=np.bool_)
    if "masks" not in result:
        result["masks"] = 1.0 - result["dones"].astype(np.float32)

    # Old chunks predate simulator snapshots. They remain valid for training,
    # but cannot be used for state-restored Q-gap evaluation.
    if "env_states" not in result:
        result["env_states"] = np.empty(size, dtype=object)
        result["env_states"][:] = None

    result = {
        key: np.asarray(result[key], dtype=np.float32)
        if key not in ("dones", "truncations", "env_states")
        else np.asarray(result[key], dtype=np.bool_)
        for key in TRANSITION_KEYS
    }
    if not all(len(value) == size for value in result.values()):
        raise ValueError(f"Transition arrays have unequal lengths: {path}")
    return result


class ChunkReplayBuffer:
    """Concurrent replay store with one immutable pickle per collector flush."""

    def __init__(self, root: str | Path, capacity: int | None = None):
        self.root = Path(root).expanduser().resolve()
        self.chunks = self.root / "chunks"
        self.chunks.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity

    def append(self, transitions: dict[str, Any], metadata: dict[str, Any] | None = None) -> Path:
        arrays = {
            key: np.asarray(transitions[key])
            for key in TRANSITION_KEYS
            if key in transitions
        }
        if "env_states" not in arrays:
            size = len(arrays["observations"])
            arrays["env_states"] = np.empty(size, dtype=object)
            arrays["env_states"][:] = None
        missing = set(TRANSITION_KEYS) - set(arrays)
        if missing:
            raise KeyError(f"Replay chunk is missing required keys: {sorted(missing)}")
        size = len(arrays["observations"])
        if not size or not all(len(value) == size for value in arrays.values()):
            raise ValueError("Replay chunk arrays must be non-empty and equally sized.")
        payload = {**arrays, "metadata": metadata or {}, "created_at": time.time()}
        name = f"{time.time_ns():020d}_{os.getpid()}_{uuid.uuid4().hex}.pkl"
        return atomic_pickle_dump(payload, self.chunks / name)

    def snapshot_paths(self) -> list[Path]:
        return sorted(self.chunks.glob("*.pkl"))

    def load_snapshot(self, paths: Iterable[str | Path] | None = None) -> dict[str, np.ndarray]:
        selected = self.snapshot_paths() if paths is None else [Path(path) for path in paths]
        pieces: dict[str, list[np.ndarray]] = {key: [] for key in TRANSITION_KEYS}
        for path in selected:
            chunk = load_transition_data(path)
            for key in TRANSITION_KEYS:
                pieces[key].append(chunk[key])
        if not selected:
            raise ValueError(f"Replay buffer is empty: {self.root}")
        merged = {key: np.concatenate(values, axis=0) for key, values in pieces.items()}
        if self.capacity is not None and len(merged["rewards"]) > self.capacity:
            merged = {key: value[-self.capacity :] for key, value in merged.items()}
        return merged

    def size(self) -> int:
        total = 0
        for path in self.snapshot_paths():
            with path.open("rb") as file:
                chunk = pickle.load(file)
            total += len(chunk["rewards"])
        return total


class VersionManager:
    """Publish and discover immutable encoder, decoder, and policy versions."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def component_dir(self, component: str, version: int) -> Path:
        return self.root / f"{component}_{version}"

    def component_checkpoint(self, component: str, version: int) -> Path:
        return self.component_dir(component, version) / "checkpoint.pkl"

    def is_component_ready(self, component: str, version: int) -> bool:
        directory = self.component_dir(component, version)
        return (directory / "READY").is_file() and (directory / "checkpoint.pkl").is_file()

    def publish_component(
        self,
        component: str,
        version: int,
        checkpoint_source: str | Path,
        metadata: dict[str, Any],
    ) -> Path:
        final = self.component_dir(component, version)
        if final.exists():
            if self.is_component_ready(component, version):
                raise FileExistsError(f"Version already published: {final}")
            raise RuntimeError(f"Incomplete version directory exists: {final}")
        temporary = self.root / f".{component}_{version}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            shutil.copy2(Path(checkpoint_source).expanduser(), temporary / "checkpoint.pkl")
            atomic_json_dump(metadata, temporary / "metadata.json")
            (temporary / "READY").write_text("ready\n", encoding="utf-8")
            os.replace(temporary, final)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return final / "checkpoint.pkl"

    def publish_policy(self, version: int) -> Path:
        if not self.is_component_ready("encoder", version):
            raise RuntimeError(f"encoder_{version} is not READY")
        if not self.is_component_ready("decoder", version):
            raise RuntimeError(f"decoder_{version} is not READY")
        final = self.component_dir("policy", version)
        if final.exists():
            checkpoint = final / "checkpoint.pkl"
            if not checkpoint.is_file():
                atomic_pickle_dump(
                    self._combined_policy_checkpoint(version), checkpoint
                )
            return final
        temporary = self.root / f".policy_{version}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir()
        metadata = {
            "version": version,
            "encoder_checkpoint": str(self.component_checkpoint("encoder", version)),
            "decoder_checkpoint": str(self.component_checkpoint("decoder", version)),
            "created_at": time.time(),
        }
        atomic_pickle_dump(
            self._combined_policy_checkpoint(version), temporary / "checkpoint.pkl"
        )
        atomic_json_dump(metadata, temporary / "metadata.json")
        (temporary / "READY").write_text("ready\n", encoding="utf-8")
        os.replace(temporary, final)
        return final

    def _combined_policy_checkpoint(self, version: int) -> dict[str, Any]:
        """Build one self-contained Encoder_n + Decoder_n evaluation artifact."""
        encoder_path = self.component_checkpoint("encoder", version)
        decoder_path = self.component_checkpoint("decoder", version)
        with encoder_path.open("rb") as file:
            encoder = pickle.load(file)
        with decoder_path.open("rb") as file:
            decoder = pickle.load(file)
        if not isinstance(encoder, dict) or not isinstance(decoder, dict):
            raise TypeError("Encoder and decoder checkpoints must be dictionaries.")

        encoder_config = encoder.get("rlpd_encoder_config", encoder.get("config"))
        if encoder_config is None:
            raise KeyError("Encoder checkpoint is missing its RLPD config.")

        # Start with the complete encoder train state, then make Decoder_n the
        # canonical decoder at the standard top-level fields. In particular,
        # overwrite fm_params embedded by Encoder_n, which describe the fixed
        # Decoder_{n-1} used during encoder training rather than Policy_n.
        combined = dict(encoder)
        combined.update(decoder)
        combined.update({
            "checkpoint_format": "gorl_online_rlpd_fm_policy",
            "checkpoint_version": 1,
            "policy_version": version,
            "rlpd_encoder_config": encoder_config,
            "fm_params": decoder["params"],
            "fm_obs_stats": decoder["obs_stats"],
            "encoder_checkpoint": str(encoder_path),
            "decoder_checkpoint": str(decoder_path),
        })
        return combined

    def ready_policy_versions(self) -> list[int]:
        versions = []
        for path in self.root.glob("policy_*/READY"):
            try:
                version = int(path.parent.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if self.is_component_ready("encoder", version) and self.is_component_ready("decoder", version):
                versions.append(version)
        return sorted(versions)

    def latest_policy(self) -> tuple[int, Path, Path] | None:
        versions = self.ready_policy_versions()
        if not versions:
            return None
        version = versions[-1]
        return (
            version,
            self.component_checkpoint("encoder", version),
            self.component_checkpoint("decoder", version),
        )

    def wait_component(self, component: str, version: int, poll_seconds: float, stop_file: Path | None = None) -> Path:
        while not self.is_component_ready(component, version):
            if stop_file is not None and stop_file.exists():
                raise InterruptedError("Pipeline stop requested.")
            time.sleep(poll_seconds)
        return self.component_checkpoint(component, version)
