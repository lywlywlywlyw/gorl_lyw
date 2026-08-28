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
            if target == "env_states":
                # Simulator states are heterogeneous dictionaries; constructing
                # the object array element-by-element prevents NumPy from trying
                # to broadcast nested qpos/qvel arrays.
                env_states = np.empty(len(value), dtype=object)
                for index, state in enumerate(value):
                    env_states[index] = state
                result[target] = env_states
            else:
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

    # ``env_states`` contains dictionaries with heterogeneous MuJoCo arrays.
    # Never coerce it to bool (the previous implementation accidentally did so
    # by grouping it with ``dones`` / ``truncations``), and avoid NumPy trying to
    # recursively broadcast dictionary values into a multidimensional array.
    env_states = np.empty(size, dtype=object)
    for index, state in enumerate(result["env_states"]):
        env_states[index] = state
    result = {
        key: np.asarray(result[key], dtype=np.float32)
        if key not in ("dones", "truncations", "env_states")
        else (
            np.asarray(result[key], dtype=np.bool_)
            if key in ("dones", "truncations")
            else env_states
        )
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

    def latest_component(self, component: str) -> tuple[int, Path] | None:
        versions = []
        for path in self.root.glob(f"{component}_*/READY"):
            try:
                version = int(path.parent.name.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if self.is_component_ready(component, version):
                versions.append(version)
        if not versions:
            return None
        version = max(versions)
        return version, self.component_checkpoint(component, version)

    def publish_policy(
        self,
        version: int,
        encoder_version: int | None = None,
        decoder_version: int | None = None,
    ) -> Path:
        encoder_version = version if encoder_version is None else encoder_version
        decoder_version = version if decoder_version is None else decoder_version
        if not self.is_component_ready("encoder", encoder_version):
            raise RuntimeError(f"encoder_{encoder_version} is not READY")
        if not self.is_component_ready("decoder", decoder_version):
            raise RuntimeError(f"decoder_{decoder_version} is not READY")
        final = self.component_dir("policy", version)
        if final.exists():
            checkpoint = final / "checkpoint.pkl"
            if not checkpoint.is_file():
                atomic_pickle_dump(
                    self._combined_policy_checkpoint(
                        version, encoder_version, decoder_version
                    ),
                    checkpoint,
                )
            return final
        temporary = self.root / f".policy_{version}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir()
        metadata = {
            "version": version,
            "encoder_version": encoder_version,
            "decoder_version": decoder_version,
            "encoder_checkpoint": str(
                self.component_checkpoint("encoder", encoder_version)
            ),
            "decoder_checkpoint": str(
                self.component_checkpoint("decoder", decoder_version)
            ),
            "created_at": time.time(),
        }
        atomic_pickle_dump(
            self._combined_policy_checkpoint(
                version, encoder_version, decoder_version
            ),
            temporary / "checkpoint.pkl",
        )
        atomic_json_dump(metadata, temporary / "metadata.json")
        (temporary / "READY").write_text("ready\n", encoding="utf-8")
        os.replace(temporary, final)
        return final

    def _combined_policy_checkpoint(
        self, version: int, encoder_version: int, decoder_version: int
    ) -> dict[str, Any]:
        """Build one self-contained policy from an explicit encoder/decoder pair."""
        encoder_path = self.component_checkpoint("encoder", encoder_version)
        decoder_path = self.component_checkpoint("decoder", decoder_version)
        with encoder_path.open("rb") as file:
            encoder = pickle.load(file)
        with decoder_path.open("rb") as file:
            decoder = pickle.load(file)
        if not isinstance(encoder, dict) or not isinstance(decoder, dict):
            raise TypeError("Encoder and decoder checkpoints must be dictionaries.")

        encoder_config = encoder.get("rlpd_encoder_config", encoder.get("config"))
        if encoder_config is None:
            raise KeyError("Encoder checkpoint is missing its RLPD config.")

        # Start with the complete encoder train state, then make the decoder
        # frozen for that encoder stage canonical at the standard top-level fields.
        combined = dict(encoder)
        combined.update(decoder)
        combined.update({
            "checkpoint_format": "gorl_online_rlpd_fm_policy",
            "checkpoint_version": 1,
            "policy_version": version,
            "encoder_version": encoder_version,
            "decoder_version": decoder_version,
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
            if self.policy_components(version) is not None:
                versions.append(version)
        return sorted(versions)

    def policy_components(self, version: int) -> tuple[Path, Path] | None:
        directory = self.component_dir("policy", version)
        if not (directory / "READY").is_file():
            return None
        metadata_path = directory / "metadata.json"
        checkpoint_path = directory / "checkpoint.pkl"
        if not metadata_path.is_file() or not checkpoint_path.is_file():
            return None
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        encoder_version = int(metadata.get("encoder_version", version))
        decoder_version = int(metadata.get("decoder_version", version))
        if not self.is_component_ready("encoder", encoder_version):
            return None
        if not self.is_component_ready("decoder", decoder_version):
            return None
        return (
            self.component_checkpoint("encoder", encoder_version),
            self.component_checkpoint("decoder", decoder_version),
        )

    def latest_policy(self) -> tuple[int, Path, Path] | None:
        versions = self.ready_policy_versions()
        if not versions:
            return None
        version = versions[-1]
        components = self.policy_components(version)
        assert components is not None
        return version, *components

    def wait_policy(
        self,
        version: int,
        poll_seconds: float,
        stop_file: Path | None = None,
    ) -> tuple[Path, Path]:
        while True:
            components = self.policy_components(version)
            if components is not None:
                return components
            if stop_file is not None and stop_file.exists():
                raise InterruptedError("Pipeline stop requested.")
            time.sleep(poll_seconds)

    def wait_component(self, component: str, version: int, poll_seconds: float, stop_file: Path | None = None) -> Path:
        while not self.is_component_ready(component, version):
            if stop_file is not None and stop_file.exists():
                raise InterruptedError("Pipeline stop requested.")
            time.sleep(poll_seconds)
        return self.component_checkpoint(component, version)
