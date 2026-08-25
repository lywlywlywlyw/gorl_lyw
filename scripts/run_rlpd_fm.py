"""Asynchronous online training pipeline for an RLPD encoder and FM decoder."""

import datetime
import json
import os
import pickle

# Inherited by the parent and every spawned worker before any JAX import.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import sys
import time
import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np

# Support ``python scripts/run_rlpd_fm.py`` without
# requiring callers to configure PYTHONPATH first.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import tyro
from envs.robomimic.online_config.training_config import TrainingConfig
from envs.robomimic.online_config.env_config import EnvConfig
from scripts.components.online_pipeline_ipc import (
    ChunkReplayBuffer,
    VersionManager,
    atomic_pickle_dump,
    load_transition_data,
)


def _wait_for_replay(replay: ChunkReplayBuffer, minimum: int, stop: Path, poll: float) -> None:
    while replay.size() < minimum:
        if stop.exists():
            raise InterruptedError("Pipeline stop requested.")
        time.sleep(poll)


def _write_replay_snapshot(replay: ChunkReplayBuffer, target: Path) -> Path:
    """Atomically freeze the chunk list and contents used by one trainer stage."""
    paths = replay.snapshot_paths()
    data = replay.load_snapshot(paths)
    return atomic_pickle_dump(data, target)


def _configure_worker_gpu(role: str, gpu_id: int) -> None:
    """Restrict one spawned async worker to one physical GPU.

    This must run before importing worker modules because those modules import
    JAX at module scope. After ``CUDA_VISIBLE_DEVICES`` is set, the selected
    physical GPU is exposed to that worker as local device 0.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    # EGL enumerates physical devices independently of CUDA's local numbering.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
    print(
        f"[{role}] physical GPU {gpu_id} selected "
        "(CUDA_VISIBLE_DEVICES; local device 0).",
        flush=True,
    )


def _encoder_worker(settings: dict) -> None:
    _configure_worker_gpu("encoder", settings["encoder_gpu_id"])
    from scripts.components.train_encoder_rlpd import train_async_stage

    manager = VersionManager(settings["versions_dir"])
    replay = ChunkReplayBuffer(settings["replay_dir"], settings["replay_capacity"])
    stop = Path(settings["stop_file"])
    for version in range(settings["start_version"], settings["max_version"] + 1):
        decoder = manager.wait_component("decoder", version - 1, settings["poll_seconds"], stop)
        previous_encoder = manager.wait_component("encoder", version - 1, settings["poll_seconds"], stop)
        _wait_for_replay(replay, settings["minimum_replay_size"], stop, settings["poll_seconds"])
        snapshot = Path(settings["snapshots_dir"]) / f"encoder_{version}_replay.pkl"
        _write_replay_snapshot(replay, snapshot)
        temporary_output = Path(settings["work_dir"]) / f"encoder_{version}.pkl"
        train_async_stage(
            decoder_checkpoint_path=str(decoder),
            previous_encoder_checkpoint_path=str(previous_encoder),
            demo_buffer_path=settings["demo_buffer_path"],
            replay_snapshot_path=str(snapshot),
            output_checkpoint_path=str(temporary_output),
            version=version,
            train_env_steps=settings["encoder_train_env_steps"],
            demo_ratio=settings["encoder_demo_ratio"],
            replay_ratio=settings["encoder_replay_ratio"],
            metrics_file=settings["metrics_file"],
            inherit_optimizer_state=version > 1,
        )
        manager.publish_component("encoder", version, temporary_output, {
            "version": version,
            "fixed_decoder_version": version - 1,
            "replay_snapshot": str(snapshot),
            "demo_ratio": settings["encoder_demo_ratio"],
            "replay_ratio": settings["encoder_replay_ratio"],
            "train_env_steps": settings["encoder_train_env_steps"],
            "inherited_optimizer_state": version > 1,
        })
        temporary_output.unlink(missing_ok=True)


def _decoder_worker(settings: dict) -> None:
    _configure_worker_gpu("decoder", settings["decoder_gpu_id"])
    from scripts.components.train_decoder_fm import train_async_stage

    manager = VersionManager(settings["versions_dir"])
    replay = ChunkReplayBuffer(settings["replay_dir"], settings["replay_capacity"])
    stop = Path(settings["stop_file"])
    for version in range(settings["start_version"], settings["max_version"] + 1):
        encoder = manager.wait_component("encoder", version, settings["poll_seconds"], stop)
        previous_decoder = manager.wait_component("decoder", version - 1, settings["poll_seconds"], stop)
        _wait_for_replay(replay, settings["minimum_replay_size"], stop, settings["poll_seconds"])
        snapshot = Path(settings["snapshots_dir"]) / f"decoder_{version}_replay.pkl"
        _write_replay_snapshot(replay, snapshot)
        temporary_output = Path(settings["work_dir"]) / f"decoder_{version}.pkl"
        train_async_stage(
            encoder_checkpoint_path=str(encoder),
            previous_decoder_checkpoint_path=str(previous_decoder),
            replay_snapshot_path=str(snapshot),
            output_checkpoint_path=str(temporary_output),
            version=version,
            train_steps=settings["decoder_train_steps"],
            metrics_file=settings["metrics_file"],
            inherit_optimizer_state=version > 1,
        )
        manager.publish_component("decoder", version, temporary_output, {
            "version": version,
            "fixed_encoder_version": version,
            "previous_decoder_version": version - 1,
            "replay_snapshot": str(snapshot),
            "train_steps": settings["decoder_train_steps"],
            "inherited_optimizer_state": version > 1,
        })
        manager.publish_policy(version)
        temporary_output.unlink(missing_ok=True)


def _collector_worker(settings: dict) -> None:
    _configure_worker_gpu("collector", settings["collector_gpu_id"])
    from scripts.components.collect_data_fm import run_async_collector

    run_async_collector(
        pipeline_root=settings["versions_dir"],
        replay_buffer_dir=settings["replay_dir"],
        stop_file=settings["stop_file"],
        poll_seconds=settings["poll_seconds"],
        rollout_steps=settings["collector_rollout_steps"],
        replay_capacity=settings["replay_capacity"],
        metrics_file=settings["metrics_file"],
        evaluation_dir=settings["evaluation_dir"],
    )


def _bootstrap_async_pipeline(
    manager: VersionManager,
    initial_checkpoint: Path,
) -> None:
    """Publish immutable encoder_0 + decoder_0 extracted from one offline checkpoint."""
    with initial_checkpoint.open("rb") as file:
        checkpoint = pickle.load(file)
    encoder_checkpoint = dict(checkpoint)
    # Combined offline checkpoints keep the decoder config in the conventional
    # ``config`` field and the RLPD config in a dedicated field.  Each async
    # component, however, expects ``config`` to describe that component.
    encoder_checkpoint["config"] = checkpoint["rlpd_encoder_config"]
    bootstrap_dir = manager.root / ".bootstrap"
    bootstrap_dir.mkdir(exist_ok=True)
    encoder_source = atomic_pickle_dump(
        encoder_checkpoint, bootstrap_dir / "encoder_0.pkl"
    )
    decoder_source = atomic_pickle_dump(checkpoint, bootstrap_dir / "decoder_0.pkl")
    if not manager.is_component_ready("encoder", 0):
        manager.publish_component("encoder", 0, encoder_source, {
            "version": 0, "source": str(initial_checkpoint), "bootstrap": True,
        })
    if not manager.is_component_ready("decoder", 0):
        manager.publish_component("decoder", 0, decoder_source, {
            "version": 0, "source": str(initial_checkpoint), "bootstrap": True,
        })
    manager.publish_policy(0)
    shutil.rmtree(bootstrap_dir, ignore_errors=True)


def run_async_pipeline(
    offline_checkpoint_path: str,
    demo_buffer_path: str,
    run_dir: str | None = None,
    num_versions: int = 24,
    encoder_train_env_steps: int = 1000,
    decoder_train_steps: int = 500,
    encoder_demo_ratio: float = 0.5,
    encoder_replay_ratio: float = 0.5,
    minimum_replay_size: int = 49152,
    replay_capacity: int | None = 200000,
    collector_rollout_steps: int = 32,
    poll_seconds: float = 1.0,
    collector_gpu_id: int = 0,
    encoder_gpu_id: int = 0,
    decoder_gpu_id: int = 0,
    parent_gpu_id: int = 0,
) -> None:
    """Run Collector, Encoder Trainer and Decoder Trainer as independent processes."""
    if not np.isclose(encoder_demo_ratio + encoder_replay_ratio, 1.0):
        raise ValueError("encoder demo/replay ratios must sum to 1.0")
    if num_versions < 1:
        raise ValueError("num_versions must be at least 1")
    if minimum_replay_size < 1:
        raise ValueError("minimum_replay_size must be positive")
    worker_gpu_ids = {
        "collector": collector_gpu_id,
        "encoder": encoder_gpu_id,
        "decoder": decoder_gpu_id,
    }
    invalid_gpu_ids = {
        role: gpu_id for role, gpu_id in worker_gpu_ids.items() if gpu_id < 0
    }
    if invalid_gpu_ids:
        raise ValueError(f"async worker GPU IDs must be non-negative: {invalid_gpu_ids}")
    if parent_gpu_id < 0:
        raise ValueError("parent_gpu_id must be non-negative")

    # The parent imports JAX below to validate environment dimensions. It may
    # share a physical GPU with every worker; preallocation is disabled globally.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(parent_gpu_id)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(parent_gpu_id)
    config = TrainingConfig().to_dict() | EnvConfig().to_dict()
    from envs.robomimic.RobomimicEnv import RobomimicEnv
    env = RobomimicEnv(dataset_path=config["dataset_path"], reward_shaping=config["dense_reward"])
    obs_dim = int(env.observation_size)
    action_dim = int(env.action_size)
    checkpoint = _validate_offline_checkpoint(
        Path(offline_checkpoint_path), obs_dim, action_dim,
        config["env_name"], require_rlpd_state=True,
    )
    env.close()
    # Validate and canonicalize the successful demonstrations before processes start.
    demo = load_transition_data(demo_buffer_path)
    if demo["observations"].shape[-1] != obs_dim:
        raise ValueError("demo_buffer observation dimension does not match environment")
    if demo["actions"].shape[-1] != action_dim:
        raise ValueError("demo_buffer action dimension does not match environment")
    root = Path(run_dir) if run_dir else Path("results") / (
        f"rlpd_fm_async_{config['env_name']}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    )
    root.mkdir(parents=True, exist_ok=True)
    canonical_demo = atomic_pickle_dump(demo, root / "buffers" / "demo_buffer.pkl")
    versions = root / "versions"
    replay_dir = root / "buffers" / "replay_buffer"
    snapshots = root / "snapshots"
    work = root / "work"
    for directory in (versions, replay_dir, snapshots, work):
        directory.mkdir(parents=True, exist_ok=True)
    evaluation_dir = root / "evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    stop = root / "STOP"
    stop.unlink(missing_ok=True)
    manager = VersionManager(versions)
    _bootstrap_async_pipeline(manager, checkpoint)
    settings = {
        "versions_dir": str(versions), "replay_dir": str(replay_dir),
        "snapshots_dir": str(snapshots), "work_dir": str(work),
        "stop_file": str(stop), "demo_buffer_path": str(canonical_demo),
        "metrics_file": str(root / "async_metrics.jsonl"),
        "evaluation_dir": str(evaluation_dir),
        "start_version": 1, "max_version": num_versions,
        "encoder_train_env_steps": encoder_train_env_steps,
        "decoder_train_steps": decoder_train_steps,
        "encoder_demo_ratio": encoder_demo_ratio,
        "encoder_replay_ratio": encoder_replay_ratio,
        "minimum_replay_size": minimum_replay_size,
        "replay_capacity": replay_capacity, "collector_rollout_steps": collector_rollout_steps,
        "poll_seconds": poll_seconds,
        "collector_gpu_id": collector_gpu_id,
        "encoder_gpu_id": encoder_gpu_id,
        "decoder_gpu_id": decoder_gpu_id,
        "parent_gpu_id": parent_gpu_id,
    }
    atomic_pickle_dump(settings, root / "pipeline_settings.pkl")
    print(
        "Asynchronous GPU assignment: "
        f"collector=cuda:{collector_gpu_id}, "
        f"encoder=cuda:{encoder_gpu_id}, decoder=cuda:{decoder_gpu_id}; "
        f"parent validation=cuda:{parent_gpu_id}"
    )
    metrics_file = Path(settings["metrics_file"])
    run_id = root.name
    wandb_run = None
    if config["wandb_enabled"]:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("wandb is required when wandb_enabled=True.") from error
        wandb_run = wandb.init(
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            name=run_id,
            id=run_id,
            group=config["wandb_group"] or run_id,
            tags=list(config["wandb_tags"]),
            mode=config["wandb_mode"],
            config={
                **config,
                **settings,
                "pipeline_mode": "async",
                "offline_checkpoint_path": str(checkpoint),
            },
        )
        wandb_run.define_metric("pipeline/version")
        for namespace in ("collector", "encoder", "decoder", "eval", "video"):
            wandb_run.define_metric(f"{namespace}/*", step_metric="pipeline/version")
        wandb_run.log({"pipeline/version": 0, "pipeline/started": 1})
    metrics_offset = 0
    context = mp.get_context("spawn")
    processes = [
        context.Process(name="data-collector", target=_collector_worker, args=(settings,)),
        context.Process(name="encoder-trainer", target=_encoder_worker, args=(settings,)),
        context.Process(name="decoder-trainer", target=_decoder_worker, args=(settings,)),
    ]
    for process in processes:
        process.start()
    pipeline_error: BaseException | None = None
    try:
        while any(process.is_alive() for process in processes):
            failed = next((p for p in processes if p.exitcode not in (None, 0)), None)
            if failed is not None:
                raise RuntimeError(f"{failed.name} exited with code {failed.exitcode}")
            # Trainers finish after num_versions; collector is intentionally endless.
            if all(not p.is_alive() for p in processes[1:]):
                break
            if wandb_run is not None:
                metrics_offset = _forward_metrics(metrics_file, wandb_run, metrics_offset)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping asynchronous pipeline...")
    except BaseException as error:
        pipeline_error = error
    finally:
        stop.write_text("stop\n", encoding="utf-8")
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        if wandb_run is not None:
            _forward_metrics(metrics_file, wandb_run, metrics_offset)
    failures = {
        p.name: p.exitcode for p in processes[1:] if p.exitcode not in (0, None)
    }
    if pipeline_error is not None or failures:
        if wandb_run is not None:
            wandb_run.log({
                "pipeline/failed": 1,
                "pipeline/failures": str(failures),
                "pipeline/error": str(pipeline_error) if pipeline_error is not None else "",
            })
            wandb_run.finish(exit_code=1)
        if pipeline_error is not None:
            raise pipeline_error
        raise RuntimeError(f"Asynchronous pipeline failed: {failures}")
    if wandb_run is not None:
        wandb_run.log({"pipeline/version": num_versions, "pipeline/completed": 1})
        wandb_run.finish()
    print(f"Asynchronous pipeline completed through Policy_{num_versions}: {root}")


def _forward_metrics(metrics_file: Path, wandb_run, offset: int) -> int:
    """Forward complete JSONL events written since offset to the parent W&B run."""
    if not metrics_file.exists():
        return offset

    with metrics_file.open("r", encoding="utf-8") as file:
        file.seek(offset)
        while True:
            line_start = file.tell()
            line = file.readline()
            if not line:
                return file.tell()
            if not line.endswith("\n"):
                return line_start

            metrics = json.loads(line)
            video_path = metrics.pop("_video_path", None)
            if video_path is not None:
                import wandb
                metrics["video/evaluation"] = wandb.Video(video_path, format="mp4")
            wandb_run.log(metrics)


def _validate_offline_checkpoint(
    checkpoint_path: Path,
    expected_obs_dim: int,
    expected_action_dim: int,
    expected_env_name: str,
    require_rlpd_state: bool = False,
) -> Path:
    """Validate an offline FM decoder checkpoint before starting the pipeline."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Offline checkpoint not found: {checkpoint_path}")

    with checkpoint_path.open("rb") as file:
        checkpoint = pickle.load(file)
    if not isinstance(checkpoint, dict):
        raise ValueError("Offline checkpoint must contain a dictionary.")

    required_fields = {"params", "obs_stats", "config", "obs_dim", "action_dim"}
    missing_fields = sorted(required_fields.difference(checkpoint))
    if missing_fields:
        raise ValueError(
            f"Offline checkpoint {checkpoint_path} is not an online-compatible "
            f"FM decoder checkpoint; missing fields: {missing_fields}. Use "
            "checkpoint_final.pkl or checkpoint_step_*.pkl produced by "
            "run_offline_fm_frozen_robomimic.py."
        )
    if checkpoint.get("decoder_type", "fm") != "fm":
        raise ValueError(f"Checkpoint decoder_type is not 'fm': {checkpoint_path}")
    if int(checkpoint["obs_dim"]) != expected_obs_dim:
        raise ValueError(
            f"Checkpoint obs_dim={checkpoint['obs_dim']} does not match online "
            f"environment obs_dim={expected_obs_dim}."
        )
    if int(checkpoint["action_dim"]) != expected_action_dim:
        raise ValueError(
            f"Checkpoint action_dim={checkpoint['action_dim']} does not match online "
            f"environment action_dim={expected_action_dim}."
        )
    checkpoint_env_name = checkpoint.get("env_name")
    if checkpoint_env_name is not None and checkpoint_env_name != expected_env_name:
        raise ValueError(
            f"Checkpoint env_name={checkpoint_env_name!r} does not match online "
            f"env_name={expected_env_name!r}."
        )
    if require_rlpd_state:
        required_rlpd_keys = {
            "rlpd_z_actor_params",
            "rlpd_z_critic_params",
            "rlpd_z_target_critic_params",
            "rlpd_z_log_temperature",
            "rlpd_z_obs_stats",
            "rlpd_encoder_config",
        }
        missing_rlpd_keys = sorted(required_rlpd_keys.difference(checkpoint))
        if missing_rlpd_keys:
            raise ValueError(
                "Offline checkpoint cannot initialize the full RLPD encoder; "
                f"missing keys: {missing_rlpd_keys}."
            )
    return checkpoint_path


def main(
    offline_checkpoint_path: str,
    demo_buffer_path: str,
    run_dir: str | None = None,
    num_versions: int = 24,
    encoder_train_env_steps: int = 250,
    decoder_train_steps: int = 500,
    encoder_demo_ratio: float = 0.5,
    encoder_replay_ratio: float = 0.5,
    minimum_replay_size: int = 49152,
    replay_capacity: int | None = 200000,
    collector_rollout_steps: int = 32,
    poll_seconds: float = 1.0,
    collector_gpu_id: int = 0,
    encoder_gpu_id: int = 0,
    decoder_gpu_id: int = 0,
    parent_gpu_id: int = 0,
) -> None:
    """Run the asynchronous RLPD encoder + FM decoder training pipeline."""
    run_async_pipeline(
        offline_checkpoint_path=offline_checkpoint_path,
        demo_buffer_path=demo_buffer_path,
        run_dir=run_dir,
        num_versions=num_versions,
        encoder_train_env_steps=encoder_train_env_steps,
        decoder_train_steps=decoder_train_steps,
        encoder_demo_ratio=encoder_demo_ratio,
        encoder_replay_ratio=encoder_replay_ratio,
        minimum_replay_size=minimum_replay_size,
        replay_capacity=replay_capacity,
        collector_rollout_steps=collector_rollout_steps,
        poll_seconds=poll_seconds,
        collector_gpu_id=collector_gpu_id,
        encoder_gpu_id=encoder_gpu_id,
        decoder_gpu_id=decoder_gpu_id,
        parent_gpu_id=parent_gpu_id,
    )


if __name__ == "__main__":
    tyro.cli(main)
