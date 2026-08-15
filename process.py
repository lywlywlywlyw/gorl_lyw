"""统计 robomimic HDF5 数据集中成功和失败的轨迹数量。

robosuite Lift 的未归一化成功奖励是 2.25，但最终奖励还会乘以
``reward_scale / 2.25``。因此成功奖励等于 ``reward_scale``；robosuite
1.4 的默认 ``reward_scale`` 是 1.0。脚本会优先读取 HDF5 的 ``env_args``
推断阈值，而不是无条件假设成功奖励为 1。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


DEFAULT_DATASETS = (
    Path("/root/GoRL/datasets/robomimic/low_dim.hdf5"),
    Path("/root/GoRL/datasets/robomimic/mg_lift_low_dim_dense.hdf5"),
    Path("/root/GoRL/datasets/robomimic/mh_lift_low_dim.hdf5"),
)


@dataclass(frozen=True)
class DatasetStats:
    path: Path
    trajectories: int
    transitions: int
    successes: int
    failures: int
    success_reward: float
    reward_scale_source: str
    done_successes: int
    inconsistent_demos: tuple[str, ...]

    @property
    def success_rate(self) -> float:
        return self.successes / self.trajectories if self.trajectories else 0.0


def demo_sort_key(name: str) -> tuple[int, int | str]:
    """按 demo 的数字编号排序，同时兼容非标准名称。"""
    suffix = name.rsplit("_", maxsplit=1)[-1]
    return (0, int(suffix)) if suffix.isdigit() else (1, name)


def infer_success_reward(data_group: h5py.Group) -> tuple[float, str]:
    """根据 robosuite Lift 元数据推断最终的成功奖励。

    Lift 的原始成功奖励为 2.25；若 ``reward_scale`` 非空，最终成功奖励
    就是 ``reward_scale``。这些数据来自 robosuite 1.x，而 1.4 中该参数
    缺省为 1.0。
    """
    raw_env_args = data_group.attrs.get("env_args")
    if raw_env_args is None:
        raise ValueError("HDF5 data group 缺少 env_args，无法自动推断成功奖励")

    if isinstance(raw_env_args, bytes):
        raw_env_args = raw_env_args.decode("utf-8")
    env_args = json.loads(raw_env_args)
    env_name = env_args.get("env_name")
    if env_name != "Lift":
        raise ValueError(f"当前自动推断仅支持 Lift，实际 env_name={env_name!r}")

    env_kwargs = env_args.get("env_kwargs", {})
    if "reward_scale" in env_kwargs:
        reward_scale = env_kwargs["reward_scale"]
        if reward_scale is None:
            return 2.25, "env_args: reward_scale=None（未归一化）"
        return float(reward_scale), "env_args: reward_scale"

    return 1.0, "robosuite Lift 默认 reward_scale=1.0"


def analyze_dataset(
    path: Path, success_reward: float | None = None
) -> DatasetStats:
    """分析一个数据集；轨迹内任意奖励达到成功奖励即视为成功。"""
    if not path.is_file():
        raise FileNotFoundError(f"数据集不存在: {path}")

    successes = 0
    done_successes = 0
    transitions = 0
    inconsistent_demos: list[str] = []

    with h5py.File(path, "r") as hdf5_file:
        if "data" not in hdf5_file:
            raise KeyError(f"{path} 中不存在必需的 'data' group")

        data_group = hdf5_file["data"]
        demo_names = sorted(data_group.keys(), key=demo_sort_key)
        if success_reward is None:
            effective_success_reward, reward_scale_source = infer_success_reward(
                data_group
            )
        else:
            effective_success_reward = success_reward
            reward_scale_source = "命令行 --success-reward"

        for demo_name in demo_names:
            demo = data_group[demo_name]
            if "rewards" not in demo:
                raise KeyError(f"{path}: data/{demo_name} 缺少 'rewards'")

            rewards = np.asarray(demo["rewards"])
            transitions += int(rewards.size)
            reward_success = bool(
                np.any(
                    np.isclose(
                        rewards,
                        effective_success_reward,
                        rtol=1e-7,
                        atol=1e-8,
                    )
                )
            )
            successes += int(reward_success)

            # dones 的语义由数据转换时的 done_mode 决定，可能表示成功、
            # 轨迹结束或二者，因此这里只进行交叉检查，不用作主判据。
            if "dones" in demo:
                done_success = bool(np.any(np.asarray(demo["dones"]) > 0))
                done_successes += int(done_success)
                if reward_success != done_success:
                    inconsistent_demos.append(demo_name)

    trajectories = len(demo_names)
    return DatasetStats(
        path=path,
        trajectories=trajectories,
        transitions=transitions,
        successes=successes,
        failures=trajectories - successes,
        success_reward=effective_success_reward,
        reward_scale_source=reward_scale_source,
        done_successes=done_successes,
        inconsistent_demos=tuple(inconsistent_demos),
    )


def print_stats(stats: DatasetStats) -> None:
    print(f"\n数据集: {stats.path}")
    print(f"  轨迹总数: {stats.trajectories}")
    print(f"  时间步总数: {stats.transitions}")
    print(
        f"  成功奖励值: {stats.success_reward:g} "
        f"（{stats.reward_scale_source}）"
    )
    print(f"  成功轨迹: {stats.successes}")
    print(f"  失败轨迹: {stats.failures}")
    print(f"  成功率: {stats.success_rate:.2%}")
    print(f"  任意 done=1 的轨迹数: {stats.done_successes}")
    if stats.inconsistent_demos:
        preview = ", ".join(stats.inconsistent_demos[:10])
        suffix = " ..." if len(stats.inconsistent_demos) > 10 else ""
        print(
            f"  警告: {len(stats.inconsistent_demos)} 条轨迹的 reward 与 done "
            f"判定不一致: {preview}{suffix}"
        )
    else:
        print("  交叉检查: 本数据集的 reward 与任意 done=1 判定一致")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计 robomimic HDF5 数据集中的成功和失败轨迹。"
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        type=Path,
        default=list(DEFAULT_DATASETS),
        help="待分析的 HDF5 文件；不提供时分析任务指定的三个默认文件。",
    )
    parser.add_argument(
        "--success-reward",
        type=float,
        default=None,
        help=(
            "显式指定最终成功奖励；默认从 env_args 的 reward_scale 推断。"
        ),
    )
    return parser.parse_args()


def main(datasets: Sequence[Path], success_reward: float) -> int:
    had_error = False
    for path in datasets:
        try:
            print_stats(analyze_dataset(path, success_reward))
        except (OSError, KeyError, ValueError) as exc:
            had_error = True
            print(f"\n分析失败: {exc}")
    return int(had_error)


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(main(arguments.datasets, arguments.success_reward))