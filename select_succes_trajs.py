"""将 robomimic HDF5 中的成功轨迹导出为 RLPD demo buffer。

输出是一个 pickle transition 字典，可直接作为
``scripts/run_rlpd_fm.py --demo-buffer-path`` 的输入。轨迹只要任意一步
``dones > 0`` 就视为成功，并保留该轨迹的所有 transition。
"""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np


INPUT_PATH = Path(
    "/root/GoRL/datasets/robomimic/mg_lift_low_dim_dense.hdf5"
)
OUTPUT_PATH = Path(
    "/root/GoRL/datasets/robomimic/mg_lift_low_dim_dense_success.pkl"
)

TRANSITION_KEYS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "masks",
    "dones",
    "truncations",
)


def demo_sort_key(demo_name: str) -> tuple[int, int | str]:
    """让 demo_2 排在 demo_10 前，同时兼容非标准轨迹名称。"""
    suffix = demo_name.rsplit("_", maxsplit=1)[-1]
    if suffix.isdigit():
        return (0, int(suffix))
    return (1, demo_name)


def is_successful(demo_group: h5py.Group) -> bool:
    """根据 dones 判断一条轨迹是否曾经成功。"""
    if "dones" not in demo_group:
        raise KeyError(f"轨迹 {demo_group.name} 中缺少 'dones' 数据集")

    return bool(np.any(np.asarray(demo_group["dones"]) > 0))


def flatten_observations(
    obs_group: h5py.Group, obs_keys: list[str]
) -> np.ndarray:
    """按 RobomimicEnv 使用的 HDF5 键顺序展平低维 observation。"""
    missing_keys = [key for key in obs_keys if key not in obs_group]
    if missing_keys:
        raise KeyError(f"{obs_group.name} 缺少 observation 字段: {missing_keys}")

    arrays = [np.asarray(obs_group[key], dtype=np.float32) for key in obs_keys]
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValueError(f"{obs_group.name} 中 observation 数组长度不一致")
    return np.concatenate(
        [array.reshape(len(array), -1) for array in arrays], axis=-1
    )


def load_demo_transitions(
    demo_group: h5py.Group, obs_keys: list[str]
) -> dict[str, np.ndarray]:
    """将一条 robomimic 轨迹转换为标准 transition 字典。"""
    if "obs" not in demo_group or "actions" not in demo_group:
        raise KeyError(f"轨迹 {demo_group.name} 缺少 'obs' 或 'actions'")

    observations = flatten_observations(demo_group["obs"], obs_keys)
    actions = np.asarray(demo_group["actions"], dtype=np.float32)
    size = len(actions)
    rewards = np.asarray(
        demo_group["rewards"] if "rewards" in demo_group else np.zeros(size),
        dtype=np.float32,
    ).reshape(-1)
    dones = np.asarray(demo_group["dones"], dtype=np.bool_).reshape(-1)

    if "next_obs" in demo_group:
        next_observations = flatten_observations(
            demo_group["next_obs"], obs_keys
        )
    else:
        next_observations = (
            np.concatenate([observations[1:], observations[-1:]], axis=0)
            if size
            else observations.copy()
        )
        if size:
            dones[-1] = True

    transitions = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "next_observations": next_observations,
        "masks": 1.0 - dones.astype(np.float32),
        "dones": dones,
        # HDF5 没有单独的 timeout 标记；成功终止由 dones 表示。
        "truncations": np.zeros(size, dtype=np.bool_),
    }
    lengths = {key: len(value) for key, value in transitions.items()}
    if set(lengths.values()) != {size}:
        raise ValueError(f"轨迹 {demo_group.name} transition 长度不一致: {lengths}")
    return transitions


def atomic_pickle_dump(value: Any, output_path: Path) -> None:
    """原子写入 pickle，避免中断时留下不完整目标文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as file:
            pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def export_successful_demos(
    input_path: Path, output_path: Path
) -> tuple[int, int, dict[str, np.ndarray]]:
    """导出成功轨迹，并返回（轨迹数, transition 数, buffer）。"""
    input_path = input_path.expanduser()
    output_path = output_path.expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入数据集不存在: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("输入路径和输出路径不能相同")
    if output_path.suffix.lower() != ".pkl":
        raise ValueError(f"输出路径必须使用 .pkl 后缀: {output_path}")

    successful_demos: list[dict[str, np.ndarray]] = []
    with h5py.File(input_path, "r") as dataset:
        if "data" not in dataset:
            raise KeyError("输入 HDF5 文件中缺少根组 'data'")
        demo_names = sorted(dataset["data"].keys(), key=demo_sort_key)
        if not demo_names:
            raise ValueError(f"输入 HDF5 文件没有轨迹: {input_path}")

        first_demo = dataset["data"][demo_names[0]]
        if "obs" not in first_demo:
            raise KeyError(f"轨迹 {first_demo.name} 缺少 'obs' 组")
        # RobomimicEnv.load_dataset 同样直接使用首条轨迹 obs 组的键顺序。
        obs_keys = list(first_demo["obs"].keys())

        for demo_name in demo_names:
            demo_group = dataset["data"][demo_name]
            if not isinstance(demo_group, h5py.Group):
                continue
            if is_successful(demo_group):
                successful_demos.append(
                    load_demo_transitions(demo_group, obs_keys)
                )

    if not successful_demos:
        raise ValueError(f"数据集中没有成功轨迹: {input_path}")

    buffer = {
        key: np.concatenate([demo[key] for demo in successful_demos], axis=0)
        for key in TRANSITION_KEYS
    }
    total_samples = len(buffer["observations"])
    if not all(len(value) == total_samples for value in buffer.values()):
        raise ValueError("合并后的 transition 数组长度不一致")

    atomic_pickle_dump(buffer, output_path)
    return len(successful_demos), total_samples, buffer


def main() -> None:
    successful_count, total_samples, buffer = export_successful_demos(
        INPUT_PATH, OUTPUT_PATH
    )
    print(f"输入文件: {INPUT_PATH}")
    print(f"输出文件: {OUTPUT_PATH}")
    print(f"成功轨迹数: {successful_count}")
    print(f"成功轨迹 transition 总数: {total_samples}")
    print(f"observation shape: {buffer['observations'].shape}")
    print(f"action shape: {buffer['actions'].shape}")


if __name__ == "__main__":
    main()
