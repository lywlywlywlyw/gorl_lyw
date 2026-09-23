"""Evaluate sparse policy versions and log only normalized return to W&B.

Example::

    source /root/.bashrc
    conda run -n gorl_robomimic python scripts/evaluate_sparse_wandb.py \
        --pipeline-root results/rlpd_fm_async_20260923_172411/versions

The script evaluates policy versions 0, 10, 20, ... that are already fully
published.  ``--watch`` can be used to wait for later versions while the
online run is still publishing checkpoints.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import wandb


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from envs.robomimic.online_config.training_config import TrainingConfig  # noqa: E402
from scripts.components.collect_data_fm import (  # noqa: E402
    _load_policy_pair,
    _make_runtime_env,
    _record_policy_evaluation,
    _runtime_env_config,
)
from scripts.components.online_pipeline_ipc import VersionManager  # noqa: E402
from flow_policy.rollout_encoder import BatchedRolloutStateEncoderFM  # noqa: E402


def _load_run_settings(pipeline_root: Path) -> dict:
    """Load the saved pipeline settings when this is an online run."""
    settings_path = pipeline_root.parent / "pipeline_settings.pkl"
    if settings_path.is_file():
        with settings_path.open("rb") as file:
            return dict(pickle.load(file))
    return {}


def _make_config(settings: dict, environment: str | None, dataset_path: str | None,
                 eval_num_envs: int | None) -> dict:
    environment = environment or settings.get("environment", "robomimic")
    EnvConfig = _runtime_env_config(environment)
    config = TrainingConfig().to_dict() | EnvConfig().to_dict()
    config.update({key: value for key, value in settings.items() if key in config})
    config["environment"] = environment
    if dataset_path is not None:
        config["dataset_path"] = dataset_path
    if eval_num_envs is not None:
        config["eval_num_envs"] = int(eval_num_envs)
    if environment == "d4rl":
        from envs.d4rl.D4RLEnv import infer_env_name

        config["env_name"] = infer_env_name(config["dataset_path"])
    config["decoder_type"] = settings.get("decoder_type", config.get("decoder_type", "flow_matching"))
    config["environment"] = environment
    return config


def _selected_versions(manager: VersionManager, interval: int, start_version: int,
                       evaluated: set[int]) -> list[int]:
    if interval < 1:
        raise ValueError("interval must be positive")
    return [
        version
        for version in manager.ready_policy_versions()
        if version >= start_version
        and (version - start_version) % interval == 0
        and version not in evaluated
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, required=True,
                        help="Run's versions directory, e.g. results/.../versions")
    parser.add_argument("--interval", type=int, default=10,
                        help="Evaluate one version every N versions (default: 10).")
    parser.add_argument("--start-version", type=int, default=0)
    parser.add_argument("--environment", choices=("d4rl", "robomimic"), default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--eval-num-envs", type=int, default=None,
                        help="Override eval_num_envs; otherwise use the run config.")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--watch", action="store_true",
                        help="Keep waiting for future versions instead of one pass.")
    parser.add_argument("--wandb-project", default="GoRL-robomimic")
    parser.add_argument("--wandb-entity", default="yuwanliu06")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()

    pipeline_root = args.pipeline_root.expanduser().resolve()
    if not pipeline_root.is_dir():
        raise FileNotFoundError(f"Pipeline versions directory does not exist: {pipeline_root}")
    settings = _load_run_settings(pipeline_root)
    config = _make_config(settings, args.environment, args.dataset_path, args.eval_num_envs)
    manager = VersionManager(pipeline_root)
    env = _make_runtime_env(config)
    eval_pool = BatchedRolloutStateEncoderFM.init(
        env,
        jax.random.key(int(config["seed"]) + 5000),
        int(config["eval_num_envs"]),
        terminate_on_success=config["terminate_on_success"],
    )
    run_name = args.wandb_name or f"sparse_eval_{pipeline_root.parent.name}"
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        mode=args.wandb_mode,
        config={
            "source_pipeline": str(pipeline_root.parent),
            "interval": args.interval,
            "start_version": args.start_version,
            "eval_num_envs": int(config["eval_num_envs"]),
            "environment": config["environment"],
            "dataset_path": config.get("dataset_path"),
        },
    )
    wandb.define_metric("policy_version")
    wandb.define_metric("normalized_return", step_metric="policy_version")
    evaluated: set[int] = set()
    try:
        while True:
            versions = _selected_versions(manager, args.interval, args.start_version, evaluated)
            if not versions and not args.watch:
                break
            for version in versions:
                components = manager.policy_components(version)
                if components is None:
                    continue
                encoder_path, decoder_path = components
                agent, apply_tanh = _load_policy_pair(encoder_path, decoder_path, env, config)
                metrics = _record_policy_evaluation(
                    agent=agent,
                    config=config,
                    version=version,
                    output_path=None,
                    apply_tanh_in_rollout=apply_tanh,
                    rollout_state=eval_pool,
                )
                normalized_return = metrics.get("eval/normalized_return")
                if normalized_return is None:
                    raise RuntimeError(
                        "normalized_return is unavailable; this sparse script currently requires a D4RL run"
                    )
                wandb.log({
                    "policy_version": int(version),
                    "normalized_return": float(normalized_return),
                }, step=int(version))
                evaluated.add(version)
                print(json.dumps({
                    "policy_version": int(version),
                    "normalized_return": float(normalized_return),
                }, ensure_ascii=False), flush=True)
            if not args.watch:
                break
            time.sleep(max(0.1, args.poll_seconds))
    finally:
        eval_pool.close()
        env.close()
        run.finish()


if __name__ == "__main__":
    main()
