"""Add GoRL ``dones`` and ``episode_ends`` labels to a D4RL HDF5 dataset.

``dones`` follows the training convention used by robomimic in this project:
it is exactly the source ``terminals`` field. ``episode_ends`` additionally
marks time-limit boundaries (``terminals | timeouts``). The source file is
never modified.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

SOURCE = Path("/root/GoRL/datasets/d4rl/halfcheetah-medium-expert-v2.hdf5")
OUTPUT = Path("/root/GoRL/datasets/d4rl/halfcheetah-medium-expert-v2_processed.hdf5")


def process(source: Path = SOURCE, output: Path = OUTPUT) -> Path:
    source = Path(source).expanduser()
    output = Path(output).expanduser()
    if source.resolve() == output.resolve():
        raise ValueError("Output must differ from the source dataset.")
    if not source.is_file():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with h5py.File(source, "r") as src, h5py.File(temporary, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            for name in src:
                src.copy(name, dst, name=name)
            if "terminals" not in src or "timeouts" not in src:
                raise KeyError("D4RL source must contain terminals and timeouts.")
            terminals = np.asarray(src["terminals"], dtype=np.bool_)
            timeouts = np.asarray(src["timeouts"], dtype=np.bool_)
            if terminals.shape != timeouts.shape:
                raise ValueError("terminals and timeouts must have identical shapes.")
            if "dones" in dst or "episode_ends" in dst:
                raise ValueError("Destination already contains generated labels.")
            dst.create_dataset("dones", data=terminals, dtype=np.bool_)
            dst.create_dataset("episode_ends", data=np.logical_or(terminals, timeouts), dtype=np.bool_)
            dst.flush()
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


if __name__ == "__main__":
    path = process()
    with h5py.File(path, "r") as data:
        print(f"wrote {path}")
        print(f"transitions={len(data['dones'])}")
        print(f"dones_true={int(np.asarray(data['dones']).sum())}")
        print(f"episode_ends_true={int(np.asarray(data['episode_ends']).sum())}")
