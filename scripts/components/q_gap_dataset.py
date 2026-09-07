"""Build a fixed, stratified restored-state bank for Q-gap evaluation."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def build_fixed_q_gap_states(
    dataset_path: str,
    output_path: str | Path,
    sample_count: int,
    seed: int,
) -> Path:
    if sample_count < 6:
        raise ValueError("q_gap_num_states must be at least 6 for six strata.")
    phases = ("early", "middle", "late")
    labels = (True, False)
    strata = [(successful, phase) for successful in labels for phase in phases]
    quotas = {
        stratum: sample_count // len(strata)
        + (index < sample_count % len(strata))
        for index, stratum in enumerate(strata)
    }
    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    with h5py.File(Path(dataset_path).expanduser(), "r") as dataset:
        episodes: dict[bool, list[str]] = {True: [], False: []}
        for demo_key, demo in dataset["data"].items():
            successful = bool(np.any(np.asarray(demo["success"])))
            episodes[successful].append(demo_key)
        for successful, phase in strata:
            candidates = episodes[successful]
            quota = quotas[(successful, phase)]
            if len(candidates) < quota:
                fallback = episodes[not successful]
                if len(candidates) + len(fallback) < quota:
                    raise ValueError(
                        f"Dataset has only {len(candidates)} episodes for "
                        f"successful={successful}, fewer than requested {quota}; "
                        f"the other category has only {len(fallback)} episodes."
                    )
                warnings.warn(
                    "Q-gap dataset has fewer than the requested "
                    f"{quota} {successful=} episodes; borrowing "
                    f"{quota - len(candidates)} episode(s) from the other "
                    "outcome category.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                primary_count = min(len(candidates), quota)
                chosen = (
                    list(rng.choice(candidates, size=primary_count, replace=False))
                    if primary_count
                    else []
                )
                chosen.extend(
                    list(rng.choice(
                        fallback,
                        size=quota - primary_count,
                        replace=False,
                    ))
                )
            else:
                chosen = rng.choice(candidates, size=quota, replace=False)
            for demo_key in chosen:
                demo = dataset[f"data/{demo_key}"]
                length = int(demo["states"].shape[0])
                boundaries = np.linspace(0, length, 4, dtype=int)
                phase_index = phases.index(phase)
                start = int(boundaries[phase_index])
                stop = max(start + 1, int(boundaries[phase_index + 1]))
                step = int(rng.integers(start, stop))
                obs = np.concatenate(
                    [np.asarray(demo["obs"][key][step]).reshape(-1) for key in demo["obs"]]
                ).astype(np.float32)
                records.append(
                    {
                        "demo_key": str(demo_key),
                        # Store the actual outcome, since a fallback demo may
                        # come from the other outcome category.
                        "successful_episode": bool(np.any(np.asarray(demo["success"]))),
                        "phase": phase,
                        "episode_step": step,
                        "observation": obs,
                        "states": np.asarray(demo["states"][step]).copy(),
                    }
                )
    rng.shuffle(records)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as file:
        pickle.dump(records, file)
    temporary.replace(target)
    return target
