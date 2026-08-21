"""从 robomimic HDF5 数据集中筛选成功轨迹。

该数据集使用每条轨迹下的 ``dones`` 数组标记任务是否成功。只要轨迹中
任意一步的 done 大于 0，就认为这条轨迹成功。成功轨迹会连同其全部数据、
子组、数据集属性和轨迹属性一起复制到新的 HDF5 文件中。
"""

from pathlib import Path

import h5py
import numpy as np


INPUT_PATH = Path(
    "/root/GoRL/datasets/robomimic/mg_lift_low_dim_dense.hdf5"
)
OUTPUT_PATH = Path(
    "/root/GoRL/datasets/robomimic/mg_lift_low_dim_dense_success.hdf5"
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

    dones = demo_group["dones"][()]
    return bool(np.any(dones > 0))


def filter_successful_demos(input_path: Path, output_path: Path) -> tuple[int, int]:
    """复制成功轨迹，并返回（成功轨迹数, 成功样本总数）。"""
    if not input_path.is_file():
        raise FileNotFoundError(f"输入数据集不存在: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("输入路径和输出路径不能相同")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)

    successful_count = 0
    total_samples = 0

    try:
        with h5py.File(input_path, "r") as source, h5py.File(
            temporary_path, "w"
        ) as destination:
            if "data" not in source:
                raise KeyError("输入 HDF5 文件中缺少根组 'data'")

            # 保留根节点和 data 组的元数据。
            destination.attrs.update(source.attrs)
            source_data = source["data"]
            destination_data = destination.create_group("data")
            destination_data.attrs.update(source_data.attrs)

            demo_names = sorted(source_data.keys(), key=demo_sort_key)
            for demo_name in demo_names:
                demo_group = source_data[demo_name]
                if not isinstance(demo_group, h5py.Group):
                    continue
                if not is_successful(demo_group):
                    continue

                # h5py 的 copy 会递归复制组、数据集及其属性，并保留原轨迹名。
                source_data.copy(
                    demo_name,
                    destination_data,
                    name=demo_name,
                    expand_soft=True,
                    expand_external=True,
                    expand_refs=True,
                )
                successful_count += 1

                copied_demo = destination_data[demo_name]
                if "num_samples" in copied_demo.attrs:
                    total_samples += int(copied_demo.attrs["num_samples"])
                elif "actions" in copied_demo:
                    total_samples += int(copied_demo["actions"].shape[0])
                else:
                    raise KeyError(
                        f"无法确定轨迹 {copied_demo.name} 的样本数量："
                        "缺少 num_samples 属性和 actions 数据集"
                    )

            # robomimic 使用 total 表示数据集中所有轨迹的时间步总数。
            destination_data.attrs["total"] = total_samples
            destination.flush()

        # 先写临时文件，成功关闭后再替换目标文件，避免留下半成品。
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return successful_count, total_samples


def main() -> None:
    successful_count, total_samples = filter_successful_demos(
        INPUT_PATH, OUTPUT_PATH
    )
    print(f"输入文件: {INPUT_PATH}")
    print(f"输出文件: {OUTPUT_PATH}")
    print(f"成功轨迹数: {successful_count}")
    print(f"成功轨迹样本总数: {total_samples}")


if __name__ == "__main__":
    main()
