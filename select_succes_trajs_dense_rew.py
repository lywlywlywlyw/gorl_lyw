"""处理 robomimic done 标签，并导出成功轨迹的 RLPD buffer。

处理规则：

1. 若轨迹中存在原始 ``dones=True``，保留到第一个 True（包含该步），
   删除后续数据；该轨迹的 ``success`` 仅最后一步为 True，并将该步
   ``reward`` 额外加 99。
2. 若原始 ``dones`` 全为 False，保留完整轨迹并把最后一步 ``dones``
   改为 True；该轨迹的 ``success`` 全为 False。
3. 从处理后的 HDF5 中选择最后一步 ``success=True`` 的轨迹，展平为
   transition 字典并保存为 pickle，可直接用作项目中的 RLPD demo buffer。
"""

from __future__ import annotations

import argparse
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
PROCESSED_PATH = Path(
    "/root/GoRL/datasets/robomimic/"
    "mg_lift_low_dim_dense_done_processed_no150.hdf5"
)
SUCCESS_OUTPUT_PATH = Path(
    "/root/GoRL/datasets/robomimic/"
    "mg_lift_low_dim_dense_done_processed_success_no150.pkl"
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


def copy_attributes(source: h5py.AttributeManager, target: h5py.AttributeManager) -> None:
    """复制全部 HDF5 attributes。"""
    for key, value in source.items():
        target[key] = value


def copy_demo_contents(
    source_group: h5py.Group,
    target_group: h5py.Group,
    original_length: int,
    kept_length: int,
) -> None:
    """递归复制轨迹内容，并截断所有以时间维为首维的数据集。"""
    copy_attributes(source_group.attrs, target_group.attrs)

    for name, source_item in source_group.items():
        if isinstance(source_item, h5py.Group):
            child_group = target_group.create_group(name)
            copy_demo_contents(
                source_item,
                child_group,
                original_length=original_length,
                kept_length=kept_length,
            )
            continue

        if source_item.ndim > 0 and source_item.shape[0] == original_length:
            data = source_item[:kept_length]
        else:
            data = source_item[()]
        target_dataset = target_group.create_dataset(name, data=data)
        copy_attributes(source_item.attrs, target_dataset.attrs)


def process_done_labels(input_path: Path, output_path: Path) -> dict[str, int]:
    """按首个原始 done 截断轨迹，并写入 done、success 和奖励加成。"""
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入数据集不存在: {input_path}")
    if input_path == output_path:
        raise ValueError("输入 HDF5 和输出 HDF5 路径不能相同")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    successful_count = 0
    failed_count = 0
    total_samples = 0

    try:
        with h5py.File(input_path, "r") as source, h5py.File(
            temporary_path, "w"
        ) as target:
            if "data" not in source or not isinstance(source["data"], h5py.Group):
                raise KeyError("输入 HDF5 文件中缺少根组 'data'")

            copy_attributes(source.attrs, target.attrs)
            for root_name, root_item in source.items():
                if root_name != "data":
                    source.copy(root_item, target, name=root_name)

            source_data = source["data"]
            target_data = target.create_group("data")
            copy_attributes(source_data.attrs, target_data.attrs)
            demo_names = sorted(source_data.keys(), key=demo_sort_key)
            if not demo_names:
                raise ValueError(f"输入 HDF5 文件没有轨迹: {input_path}")

            for demo_name in demo_names:
                source_demo = source_data[demo_name]
                if not isinstance(source_demo, h5py.Group):
                    source_data.copy(source_demo, target_data, name=demo_name)
                    continue
                if "dones" not in source_demo:
                    raise KeyError(f"轨迹 {source_demo.name} 中缺少 'dones'")
                if "rewards" not in source_demo:
                    raise KeyError(f"轨迹 {source_demo.name} 中缺少 'rewards'")

                original_dones = np.asarray(source_demo["dones"]).reshape(-1)
                original_length = len(original_dones)
                if original_length == 0:
                    raise ValueError(f"轨迹 {source_demo.name} 是空轨迹")
                if len(source_demo["rewards"]) != original_length:
                    raise ValueError(
                        f"轨迹 {source_demo.name} 的 dones 和 rewards 长度不一致"
                    )

                done_indices = np.flatnonzero(original_dones.astype(bool))
                is_success = bool(done_indices.size)
                kept_length = int(done_indices[0] + 1) if is_success else original_length

                target_demo = target_data.create_group(demo_name)
                copy_demo_contents(
                    source_demo,
                    target_demo,
                    original_length=original_length,
                    kept_length=kept_length,
                )

                # 成功或达到最大 episode 长度时，末步都是真实终止。
                processed_dones = np.asarray(target_demo["dones"])
                processed_dones[...] = 0
                processed_dones[-1] = 1
                del target_demo["dones"]
                target_dones = target_demo.create_dataset(
                    "dones", data=processed_dones.astype(source_demo["dones"].dtype)
                )
                copy_attributes(source_demo["dones"].attrs, target_dones.attrs)

                success = np.zeros(kept_length, dtype=np.bool_)
                if is_success:
                    success[-1] = True
                    processed_rewards = np.asarray(target_demo["rewards"])
                    processed_rewards[-1] += 0#150
                    del target_demo["rewards"]
                    target_rewards = target_demo.create_dataset(
                        "rewards",
                        data=processed_rewards.astype(source_demo["rewards"].dtype),
                    )
                    copy_attributes(source_demo["rewards"].attrs, target_rewards.attrs)
                    successful_count += 1
                else:
                    failed_count += 1
                target_demo.create_dataset("success", data=success)
                target_demo.attrs["num_samples"] = kept_length
                total_samples += kept_length

            target_data.attrs["total"] = total_samples
            target.flush()

        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return {
        "trajectories": successful_count + failed_count,
        "successful_trajectories": successful_count,
        "failed_trajectories": failed_count,
        "total_samples": total_samples,
    }


def flatten_observations(obs_group: h5py.Group, obs_keys: list[str]) -> np.ndarray:
    """按固定 observation 键顺序展平低维 observation。"""
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
    """把一条处理后的 robomimic 轨迹转换为 transition 字典。"""
    required_keys = ("obs", "next_obs", "actions", "dones", "success")
    missing_keys = [key for key in required_keys if key not in demo_group]
    if missing_keys:
        raise KeyError(f"轨迹 {demo_group.name} 缺少字段: {missing_keys}")

    observations = flatten_observations(demo_group["obs"], obs_keys)
    next_observations = flatten_observations(demo_group["next_obs"], obs_keys)
    actions = np.asarray(demo_group["actions"], dtype=np.float32)
    size = len(actions)
    rewards = np.asarray(
        demo_group["rewards"] if "rewards" in demo_group else np.zeros(size),
        dtype=np.float32,
    ).reshape(-1)
    dones = np.asarray(demo_group["dones"], dtype=np.bool_).reshape(-1)
    success = np.asarray(demo_group["success"], dtype=np.bool_).reshape(-1)

    transitions = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "next_observations": next_observations,
        "masks": 1.0 - dones.astype(np.float32),
        "dones": dones,
        "truncations": np.logical_and(dones, np.logical_not(success)),
    }
    lengths = {key: len(value) for key, value in transitions.items()}
    lengths["success"] = len(success)
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
    """选择末步 success=True 的轨迹，导出为扁平 transition buffer。"""
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"处理后的输入数据集不存在: {input_path}")
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
        obs_keys = list(first_demo["obs"].keys())

        for demo_name in demo_names:
            demo_group = dataset["data"][demo_name]
            if not isinstance(demo_group, h5py.Group):
                continue
            if "success" not in demo_group:
                raise KeyError(f"轨迹 {demo_group.name} 中缺少 'success'")
            success = np.asarray(demo_group["success"], dtype=np.bool_).reshape(-1)
            if success.size and bool(success[-1]):
                successful_demos.append(load_demo_transitions(demo_group, obs_keys))

    if not successful_demos:
        raise ValueError(f"数据集中没有末步 success=True 的轨迹: {input_path}")

    buffer = {
        key: np.concatenate([demo[key] for demo in successful_demos], axis=0)
        for key in TRANSITION_KEYS
    }
    total_samples = len(buffer["observations"])
    if not all(len(value) == total_samples for value in buffer.values()):
        raise ValueError("合并后的 transition 数组长度不一致")

    atomic_pickle_dump(buffer, output_path)
    return len(successful_demos), total_samples, buffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--processed-output", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--success-output", type=Path, default=SUCCESS_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = process_done_labels(args.input, args.processed_output)
    successful_count, success_samples, buffer = export_successful_demos(
        args.processed_output, args.success_output
    )

    print(f"原始输入文件: {args.input}")
    print(f"处理后 HDF5: {args.processed_output}")
    print(f"成功轨迹 PKL: {args.success_output}")
    print(f"轨迹总数: {stats['trajectories']}")
    print(f"成功轨迹数: {stats['successful_trajectories']}")
    print(f"未成功轨迹数: {stats['failed_trajectories']}")
    print(f"处理后 transition 总数: {stats['total_samples']}")
    print(f"导出的成功轨迹数: {successful_count}")
    print(f"成功轨迹 transition 总数: {success_samples}")
    print(f"observation shape: {buffer['observations'].shape}")
    print(f"action shape: {buffer['actions'].shape}")


if __name__ == "__main__":
    main()