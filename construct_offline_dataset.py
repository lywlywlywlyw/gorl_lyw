"""Construct a constrained, trajectory-level robomimic offline dataset.

The program only reads the source HDF5 file and performs trajectory selection.
It never imports or calls a policy, encoder, decoder, critic, or latent action
code.  The selected HDF5 keeps the robomimic ``data/demo_*`` layout.  For
compatibility with this repository it also writes a flattened successful-demo
pickle and a successful-only HDF5 file.

Success semantics follow the existing project convention.  If a source
trajectory has a per-transition ``success`` dataset it is used.  Otherwise the
default is ``original_done``: the first original ``done=True`` is the success
step and success is propagated through the rest of that complete trajectory,
matching ``select_succes_trajs_dense_rew_nocut.py``.  Other definitions remain
selectable explicitly when a dataset documents different semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.special import digamma, gammaln
from scipy.spatial import cKDTree


# robomimic trajectory fields are fixed by the source dataset schema.
ACTION_KEY = "actions"
REWARD_KEY = "rewards"
DONE_KEY = "dones"
SUCCESS_KEY = "success"
INPUT_PATH = Path("/root/GoRL/datasets/d4rl/halfcheetah-medium-expert-v2.hdf5")
PROCESSED_PATH = Path("/root/GoRL/datasets/robomimic/mg_can_low_dim_dense_done_processed_v141_nocut_filtered0_4.hdf5")
SUCCESS_OUTPUT_PATH = Path("/root/GoRL/datasets/robomimic/mg_can_low_dim_dense_done_processed_success_v141_nocut_filtered0_4.pkl")
SUCCESS_HDF5_OUTPUT_PATH = Path("/root/GoRL/datasets/robomimic/mg_can_low_dim_dense_done_processed_success_v141_nocut_filtered0_4.hdf5")


def demo_sort_key(name: str) -> tuple[int, int | str]:
    suffix = name.rsplit("_", 1)[-1]
    return (0, int(suffix)) if suffix.isdigit() else (1, name)


def copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flatten_obs(group: h5py.Group, keys: list[str]) -> np.ndarray:
    missing = [key for key in keys if key not in group]
    if missing:
        raise KeyError(f"{group.name} missing observation keys: {missing}")
    arrays = [np.asarray(group[key][()], dtype=np.float64) for key in keys]
    lengths = {array.shape[0] for array in arrays}
    if len(lengths) != 1:
        raise ValueError(f"observation lengths differ in {group.name}")
    return np.concatenate([array.reshape(array.shape[0], -1) for array in arrays], axis=1)


def success_from_source(
    demo: h5py.Group,
    rewards: np.ndarray,
    definition: str,
    threshold: float | None,
) -> tuple[np.ndarray, str]:
    n = len(rewards)
    if SUCCESS_KEY in demo:
        raw = np.asarray(demo[SUCCESS_KEY][()]).reshape(-1)
        if raw.size == 1:
            raw = np.repeat(raw, n)
        if raw.size != n:
            raise ValueError(f"{demo.name}/success length {raw.size} != {n}")
        return raw.astype(bool), "source success dataset"
    if definition == "error":
        raise ValueError(
            f"{demo.name} has no success field. Pass --success-definition "
            "original_done, reward_positive, or reward_threshold."
        )
    if definition == "original_done":
        if DONE_KEY not in demo:
            raise KeyError(f"{demo.name} has no {DONE_KEY} field")
        raw = np.asarray(demo[DONE_KEY][()]).reshape(-1).astype(bool)
        if len(raw) != n:
            raise ValueError(f"{demo.name}/dones length differs from rewards")
        return raw, "success iff any original done is true; propagated from first true"
    if definition == "reward_positive":
        return rewards > 0, "success iff reward > 0; propagated from first qualifying step"
    if definition == "reward_threshold":
        if threshold is None:
            raise ValueError("--success-threshold is required with reward_threshold")
        return rewards >= threshold, f"success iff reward >= {threshold}; propagated from first qualifying step"
    raise ValueError(f"unknown success definition: {definition}")


def propagate_success(labels: np.ndarray) -> tuple[bool, np.ndarray]:
    indices = np.flatnonzero(labels)
    if not len(indices):
        return False, np.zeros(len(labels), dtype=bool)
    out = np.zeros(len(labels), dtype=bool)
    out[int(indices[0]) :] = True
    return True, out


@dataclass
class Trajectory:
    name: str
    length: int
    episode_return: float
    successful: bool
    success: np.ndarray
    features: np.ndarray
    obs: np.ndarray
    actions: np.ndarray


def knn_entropy(features: np.ndarray, k: int, sample_size: int, repeats: int, seed: int) -> tuple[float, float]:
    if k < 1:
        raise ValueError("knn_k must be positive")
    if len(features) <= k:
        raise ValueError(f"entropy sample count {len(features)} must exceed k={k}")
    if sample_size > len(features):
        raise ValueError("entropy_sample_size exceeds available transitions")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    dim = features.shape[1]
    log_volume = (dim / 2.0) * math.log(math.pi) - gammaln(dim / 2.0 + 1.0)
    for _ in range(repeats):
        indices = rng.choice(len(features), sample_size, replace=False)
        sample = features[indices]
        distances = cKDTree(sample).query(sample, k=k + 1, workers=1)[0][:, -1]
        value = float(digamma(sample_size) - digamma(k) + log_volume)
        value += float(dim / sample_size * np.log(distances + 1e-12).sum())
        values.append(value)
    return float(np.mean(values)), float(np.std(values))


def normalize_features(trajectories: list[Trajectory], eps: float) -> dict[str, Any]:
    states = np.concatenate([t.obs for t in trajectories])
    actions = np.concatenate([t.actions for t in trajectories])
    state_mean, state_std = states.mean(0), states.std(0)
    action_mean, action_std = actions.mean(0), actions.std(0)
    state_keep = state_std >= eps
    action_keep = action_std >= eps
    for t in trajectories:
        state = t.obs
        s = (state[:, state_keep] - state_mean[state_keep]) / (state_std[state_keep] + eps)
        a = (t.actions[:, action_keep] - action_mean[action_keep]) / (action_std[action_keep] + eps)
        t.features = np.concatenate([s, a], axis=1)
    return {
        "state_mean": state_mean.tolist(), "state_std": state_std.tolist(),
        "action_mean": action_mean.tolist(), "action_std": action_std.tolist(),
        "state_kept_dimensions": np.flatnonzero(state_keep).tolist(),
        "action_kept_dimensions": np.flatnonzero(action_keep).tolist(),
        "removed_constant_state_dimensions": np.flatnonzero(~state_keep).tolist(),
        "removed_constant_action_dimensions": np.flatnonzero(~action_keep).tolist(),
        "epsilon": eps,
    }


def candidate_names(trajectories: list[Trajectory], args: argparse.Namespace, rng: np.random.Generator) -> list[str] | None:
    """Generate a complete-trajectory candidate, starting from successes.

    Success count is chosen from the feasible integer range whenever the
    success-ratio constraint is enabled. Within each outcome group, higher
    episode-return trajectories are preferred, with small seeded jitter to
    produce diverse candidates across trials.
    """
    target = len(trajectories) if args.target_num_trajectories is None else args.target_num_trajectories
    if target < 1 or target > len(trajectories):
        return None
    successes = [t for t in trajectories if t.successful]
    failures = [t for t in trajectories if not t.successful]
    if args.use_success_ratio:
        lower = max(0, int(math.ceil(args.success_ratio_min * target - 1e-12)))
        upper = min(target, int(math.floor(args.success_ratio_max * target + 1e-12)))
        lower = max(lower, target - len(failures))
        upper = min(upper, len(successes))
        if lower > upper:
            return None
        success_count = upper
    else:
        success_count = min(len(successes), target)
    failure_count = target - success_count
    if success_count > len(successes) or failure_count > len(failures):
        return None

    def ranked(pool: list[Trajectory], count: int) -> list[Trajectory]:
        if count == 0:
            return []
        scale = max(float(np.std([t.episode_return for t in pool])), 1.0)
        scores = {t.name: t.episode_return + float(rng.normal(0.0, 0.05 * scale)) for t in pool}
        return sorted(pool, key=lambda t: (-scores[t.name], demo_sort_key(t.name)))[:count]

    chosen = ranked(successes, success_count) + ranked(failures, failure_count)
    return [t.name for t in chosen]


def metrics(selected: list[Trajectory], args: argparse.Namespace, seed: int) -> dict[str, Any]:
    returns = np.asarray([t.episode_return for t in selected], dtype=np.float64)
    features = np.concatenate([t.features for t in selected])
    # The only entropy used by the prompt is the joint H_k(S, A).
    # Repeated fixed-size estimates are summarized by mean and standard deviation.
    entropy_mean, entropy_std = knn_entropy(features, args.knn_k, args.entropy_sample_size, args.entropy_num_repeats, seed)
    return {
        "num_trajectories": len(selected), "num_transitions": int(sum(t.length for t in selected)),
        "num_success_trajectories": int(sum(t.successful for t in selected)),
        "num_failure_trajectories": int(sum(not t.successful for t in selected)),
        # Trajectory-level ratio: successful trajectory count / selected trajectory count.
        "success_ratio": float(sum(1 for t in selected if t.successful) / len(selected)),
        "mean_episode_return": float(np.mean(returns)), "std_episode_return": float(np.std(returns)),
        "return_quantiles": {str(q): float(np.quantile(returns, q)) for q in (0, .25, .5, .75, 1)},
        "state_action_entropy_mean": entropy_mean, "state_action_entropy_std": entropy_std,
    }


def satisfies(m: dict[str, Any], args: argparse.Namespace) -> bool:
    target = args.target_num_trajectories
    budget_ok = target is None or m["num_trajectories"] == target
    success_ok = (not args.use_success_ratio or args.success_ratio_min <= m["success_ratio"] <= args.success_ratio_max)
    mean_return_ok = (not args.use_mean_return or ((args.mean_return_min is None or m["mean_episode_return"] >= args.mean_return_min) and (args.mean_return_max is None or m["mean_episode_return"] <= args.mean_return_max)))
    entropy_ok = (not args.use_state_action_entropy or m["state_action_entropy_mean"] >= args.state_action_entropy_min)
    return budget_ok and success_ok and mean_return_ok and entropy_ok


def choose(trajectories: list[Trajectory], args: argparse.Namespace) -> tuple[list[Trajectory], dict[str, Any], int]:
    by_name = {t.name: t for t in trajectories}
    best: tuple[list[Trajectory], dict[str, Any]] | None = None
    best_violation: tuple[float, list[Trajectory], dict[str, Any]] | None = None
    for trial in range(args.num_search_trials):
        names = candidate_names(trajectories, args, np.random.default_rng(args.seed + trial))
        if not names:
            continue
        selected = [by_name[n] for n in names]
        if not selected or len(np.concatenate([t.features for t in selected])) <= args.knn_k:
            continue
        m = metrics(selected, args, args.seed + trial)
        violations = [0.0]
        if args.use_success_ratio:
            violations.extend((
                args.success_ratio_min - m["success_ratio"],
                m["success_ratio"] - args.success_ratio_max,
            ))
        if args.use_mean_return:
            if args.mean_return_min is not None:
                violations.append(args.mean_return_min - m["mean_episode_return"])
            if args.mean_return_max is not None:
                violations.append(m["mean_episode_return"] - args.mean_return_max)
        if args.use_state_action_entropy:
            violations.append(args.state_action_entropy_min - m["state_action_entropy_mean"])
        violation = max(violations)
        if best_violation is None or violation < best_violation[0]:
            best_violation = (violation, selected, m)
        if satisfies(m, args) and (best is None or m["state_action_entropy_mean"] > best[1]["state_action_entropy_mean"]):
            best = (selected, m)
    if best is None:
        if best_violation:
            raise RuntimeError("No feasible subset found; closest candidate: " + json.dumps(best_violation[2], ensure_ascii=False))
        raise RuntimeError("No candidate subset could be generated")
    selected, m = best
    # Optional trajectory-level greedy improvement. Every proposal preserves the
    # complete-trajectory unit and all hard constraints before it is accepted.
    if args.num_greedy_swaps:
        selected_names = {t.name for t in selected}
        for _ in range(args.num_greedy_swaps):
            improved = False
            for outgoing in list(selected):
                for incoming in trajectories:
                    if incoming.name in selected_names:
                        continue
                    proposal = [t for t in selected if t.name != outgoing.name] + [incoming]
                    if sum(t.length for t in proposal) <= 0 or len(np.concatenate([t.features for t in proposal])) <= args.knn_k:
                        continue
                    proposal_metrics = metrics(proposal, args, args.seed)
                    if satisfies(proposal_metrics, args) and proposal_metrics["state_action_entropy_mean"] > m["state_action_entropy_mean"]:
                        selected, m = proposal
                        selected_names = {t.name for t in selected}
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
    selected.sort(key=lambda t: demo_sort_key(t.name))
    return selected, m, args.num_search_trials


def copy_demo(source: h5py.Group, target: h5py.Group, success: np.ndarray) -> None:
    source.file.copy(source, target.parent, name=target.name.rsplit("/", 1)[-1])
    # h5py copy above is used only as a complete structural copy; replace labels below.
    copied = target.parent[target.name.rsplit("/", 1)[-1]]
    n = len(success)
    if "dones" not in copied:
        raise KeyError(f"{source.name} missing dones")
    dones = np.zeros(n, dtype=np.asarray(source["dones"][()]).dtype)
    dones[-1] = 1
    del copied["dones"]
    ds = copied.create_dataset("dones", data=dones)
    copy_attrs(source["dones"].attrs, ds.attrs)
    if "success" in copied:
        del copied["success"]
    copied.create_dataset("success", data=success.astype(bool))
    copied.attrs["num_samples"] = n



def copy_hdf5_tree(source_item: h5py.Group, target_item: h5py.Group) -> None:
    """Copy a group recursively without h5py's large source.copy operation."""
    copy_attrs(source_item.attrs, target_item.attrs)
    for name, item in source_item.items():
        if isinstance(item, h5py.Group):
            child = target_item.create_group(name)
            copy_hdf5_tree(item, child)
        else:
            child = target_item.create_dataset(name, data=item[()])
            copy_attrs(item.attrs, child.attrs)


def cleanup_temporary_outputs(paths: list[Path]) -> None:
    for path in paths:
        path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)


def write_hdf5(source_path: Path, output_path: Path, selected: list[Trajectory], args: argparse.Namespace, success_only: bool = False) -> None:
    names = {t.name for t in selected if not success_only or t.successful}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    with h5py.File(source_path, "r") as source, h5py.File(tmp, "w") as target:
        copy_attrs(source.attrs, target.attrs)
        for name, item in source.items():
            if name != "data":
                if isinstance(item, h5py.Group):
                    child = target.create_group(name)
                    copy_hdf5_tree(item, child)
                else:
                    child = target.create_dataset(name, data=item[()])
                    copy_attrs(item.attrs, child.attrs)
        data = target.create_group("data")
        copy_attrs(source["data"].attrs, data.attrs)
        total = 0
        for name in sorted(names, key=demo_sort_key):
            src = source[f"data/{name}"]
            # Copy every unknown dataset/group and all attributes recursively.
            copied = data.create_group(name)
            copy_hdf5_tree(src, copied)
            n = int(len(src[ACTION_KEY]))
            dones = np.zeros(n, dtype=src[DONE_KEY].dtype); dones[-1] = 1
            del copied[DONE_KEY]; ds = copied.create_dataset(DONE_KEY, data=dones); copy_attrs(src[DONE_KEY].attrs, ds.attrs)
            if "success" in copied: del copied["success"]
            ds = copied.create_dataset("success", data=next(t.success for t in selected if t.name == name).astype(bool))
            copied.attrs["num_samples"] = n
            total += n
        data.attrs["total"] = total
    os.replace(tmp, output_path)


def write_success_pickle(path: Path, source_path: Path, selected: list[Trajectory], args: argparse.Namespace) -> None:
    buffers: dict[str, list[np.ndarray]] = {k: [] for k in ("observations", "actions", "rewards", "next_observations", "masks", "dones", "truncations")}
    with h5py.File(source_path, "r") as source:
        for t in selected:
            if not t.successful: continue
            g = source[f"data/{t.name}"]; obs = flatten_obs(g["obs"], args.observation_keys).astype(np.float32); nxt = flatten_obs(g["next_obs"], args.observation_keys).astype(np.float32)
            n = len(t.actions); dones = np.zeros(n, dtype=bool); dones[-1] = True
            buffers["observations"].append(obs); buffers["next_observations"].append(nxt); buffers["actions"].append(np.asarray(g[ACTION_KEY][()], dtype=np.float32)); buffers["rewards"].append(np.asarray(g[REWARD_KEY][()], dtype=np.float32).reshape(-1)); buffers["dones"].append(dones); buffers["masks"].append(1.0-dones.astype(np.float32)); buffers["truncations"].append(np.zeros(n, dtype=bool))
    value = {k: np.concatenate(v) for k,v in buffers.items()}
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix+".tmp")
    tmp.unlink(missing_ok=True)
    with tmp.open("wb") as f: pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-dataset-path", type=Path, default=INPUT_PATH)
    p.add_argument("--output-dataset-path", type=Path, default=PROCESSED_PATH)
    p.add_argument("--success-output-hdf5", type=Path, default=SUCCESS_HDF5_OUTPUT_PATH)
    p.add_argument("--success-output-pkl", type=Path, default=SUCCESS_OUTPUT_PATH)
    p.add_argument("--observation-keys", nargs="+", default=None, help="optional low-dimensional obs subset; default: all source obs keys")
    p.add_argument("--success-definition", choices=("error", "original_done", "reward_positive", "reward_threshold"), default="original_done"); p.add_argument("--success-threshold", type=float)
    p.add_argument("--target-num-trajectories", type=int, default=1000, help="number of complete trajectories; default: all source trajectories")
 
    p.add_argument("--use-success-ratio", action=argparse.BooleanOptionalAction, default=False, help="enforce success-ratio bounds")
    p.add_argument("--use-mean-return", action=argparse.BooleanOptionalAction, default=False, help="enforce mean episode-return bounds")
    p.add_argument("--use-state-action-entropy", action=argparse.BooleanOptionalAction, default=False, help="enforce kNN state-action entropy bound")
    p.add_argument("--success-ratio-min", type=float, default=0.0); p.add_argument("--success-ratio-max", type=float, default=1.0); p.add_argument("--mean-return-min", type=float); p.add_argument("--mean-return-max", type=float); p.add_argument("--state-action-entropy-min", type=float, default=-math.inf)
    p.add_argument("--knn-k", type=int, default=5); p.add_argument("--entropy-sample-size", type=int, default=2048); p.add_argument("--entropy-num-repeats", type=int, default=3); p.add_argument("--num-search-trials", type=int, default=100); p.add_argument("--num-greedy-swaps", type=int, default=0); p.add_argument("--seed", type=int, default=0); p.add_argument("--normalization-epsilon", type=float, default=1e-12); p.add_argument("--analyze-only", action="store_true")
    return p.parse_args()


def load_trajectories(path: Path, args: argparse.Namespace) -> tuple[list[Trajectory], list[str]]:
    trajectories: list[Trajectory] = []
    source_success_definition = "source success dataset"
    definitions_used: set[str] = set()
    source_observation_keys: list[str] | None = None
    with h5py.File(path, "r") as source:
        if "data" not in source: raise KeyError("source HDF5 has no data group")
        names = sorted([n for n in source["data"].keys() if isinstance(source["data"][n], h5py.Group)], key=demo_sort_key)
        if not names: raise ValueError("source HDF5 contains no trajectories")
        for name in names:
            g = source[f"data/{name}"]
            for key in (ACTION_KEY, REWARD_KEY, DONE_KEY, "obs", "next_obs"): 
                if key not in g: raise KeyError(f"data/{name} missing {key}")
            actions = np.asarray(g[ACTION_KEY][()], dtype=np.float64); rewards = np.asarray(g[REWARD_KEY][()], dtype=np.float64).reshape(-1); n = len(actions)
            if len(rewards) != n or len(g[DONE_KEY]) != n: raise ValueError(f"length mismatch in {name}")
            labels, source_success_definition = success_from_source(g, rewards, args.success_definition, args.success_threshold)
            definitions_used.add(source_success_definition)
            successful, success = propagate_success(labels)
            if source_observation_keys is None:
                source_observation_keys = list(g["obs"].keys()) if args.observation_keys is None else list(args.observation_keys)
            obs = flatten_obs(g["obs"], source_observation_keys)
            if len(obs) != n: raise ValueError(f"observation length mismatch in {name}")
            trajectories.append(Trajectory(name, n, float(rewards.sum()), successful, success, np.empty((n,0)), obs, actions))
    if source_observation_keys is None:
        raise ValueError("source dataset contains no trajectories")
    args.observation_keys = source_observation_keys
    return trajectories, sorted(definitions_used)


def print_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> int:
    args = parse_args(); source = args.source_dataset_path.expanduser().resolve()
    output = args.output_dataset_path.expanduser().resolve()
    if source == output: raise ValueError("source and output paths must differ")
    trajectories, definitions = load_trajectories(source, args); norm = normalize_features(trajectories, args.normalization_epsilon)
    all_features = np.concatenate([t.features for t in trajectories]); source_entropy = knn_entropy(all_features, args.knn_k, args.entropy_sample_size, args.entropy_num_repeats, args.seed)
    if args.analyze_only:
        selected = trajectories; m = metrics(selected, args, args.seed)
        successful_trajectories = [t for t in trajectories if t.successful]
        successful_metrics = metrics(successful_trajectories, args, args.seed + 100000) if successful_trajectories else None
        random_metrics = []
        # Random subset distributions require an explicit budget. The source
        # dataset metrics above are always computed, even without a budget.
        has_budget = args.target_num_trajectories is not None
        for trial in range(min(args.num_search_trials, 100) if has_budget else 0):
            names = candidate_names(trajectories, args, np.random.default_rng(args.seed + trial))
            if names:
                candidate = [next(t for t in trajectories if t.name == name) for name in names]
                if len(np.concatenate([t.features for t in candidate])) > args.knn_k:
                    random_metrics.append(metrics(candidate, args, args.seed + trial))
        distributions = {}
        for key in ("success_ratio", "mean_episode_return", "state_action_entropy_mean"):
            values = np.asarray([r[key] for r in random_metrics], dtype=np.float64)
            distributions[key] = {"count": int(values.size), "min": float(values.min()) if values.size else None, "max": float(values.max()) if values.size else None, "quantiles": {str(q): float(np.quantile(values, q)) for q in (.05, .25, .5, .75, .95)} if values.size else {}}
        print_report({"source_dataset_path": str(source), "source_trajectory_count": len(trajectories), "source_transition_count": sum(t.length for t in trajectories), "source_success_ratio": float(sum(1 for t in trajectories if t.successful) / len(trajectories)), "source_entropy": {"mean": source_entropy[0], "std": source_entropy[1]}, "metrics": m, "successful_only_metrics": successful_metrics, "successful_only_trajectory_count": len(successful_trajectories), "random_subset_distributions": distributions, "success_definition": definitions, "normalization_statistics": norm}); return 0
    selected, m, trials = choose(trajectories, args)
    success_h5 = args.success_output_hdf5 or output.with_name(output.stem + "_success.hdf5"); success_pkl = args.success_output_pkl or output.with_name(output.stem + "_success.pkl")
    try:
        write_hdf5(source, output, selected, args)
        write_hdf5(source, success_h5, selected, args, success_only=True)
        write_success_pickle(success_pkl, source, selected, args)
    except BaseException:
        cleanup_temporary_outputs([output, success_h5, success_pkl])
        raise
    # Independent post-write checks: reopen the produced file and verify boundaries,
    # rewards, terminal labels, and all hard constraints.
    with h5py.File(source, "r") as source_check, h5py.File(output, "r") as check:
        if "data" not in check: raise RuntimeError("written dataset has no data group")
        if sorted(check["data"].keys(), key=demo_sort_key) != sorted([t.name for t in selected], key=demo_sort_key):
            raise RuntimeError("written dataset does not contain exactly the selected trajectories")
        for t in selected:
            g = check[f"data/{t.name}"]; n = len(g[ACTION_KEY])
            if n != t.length or not bool(np.asarray(g[DONE_KEY][-1])) or np.asarray(g[DONE_KEY][:-1]).any():
                raise RuntimeError(f"invalid trajectory boundary/done labels in {t.name}")
            original = source_check[f"data/{t.name}/{REWARD_KEY}"][()]
            if not np.array_equal(np.asarray(g[REWARD_KEY][()]), np.asarray(original)):
                raise RuntimeError(f"rewards changed in {t.name}")
    report = {"source_dataset_path": str(source), "output_dataset_path": str(output), "source_dataset_hash": sha256_file(source), "selection_seed": args.seed, "selected_trajectory_ids": [t.name for t in selected], **m, "knn_k": args.knn_k, "entropy_feature_dimension": int(selected[0].features.shape[1]), "entropy_sample_size": args.entropy_sample_size, "entropy_num_repeats": args.entropy_num_repeats, "observation_keys": args.observation_keys, "action_key": ACTION_KEY, "normalization_statistics": norm, "success_definition": definitions, "search_trials_used": trials, "thresholds": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "all_thresholds_satisfied": satisfies(m, args), "post_write_validation": True}
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
