"""Complete alternating training pipeline for RLPD + FM.

This script automates the full training loop:
Stage 0: Init decoder → Encoder update → Collect data → Decoder update
Stage 1+: Encoder update → Collect data → Decoder update → Repeat
"""

import datetime
import json
import pickle
import shlex
import subprocess
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


def _encoder_worker(settings: dict) -> None:
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
        )
        manager.publish_component("encoder", version, temporary_output, {
            "version": version,
            "fixed_decoder_version": version - 1,
            "replay_snapshot": str(snapshot),
            "demo_ratio": settings["encoder_demo_ratio"],
            "replay_ratio": settings["encoder_replay_ratio"],
            "train_env_steps": settings["encoder_train_env_steps"],
        })
        temporary_output.unlink(missing_ok=True)


def _decoder_worker(settings: dict) -> None:
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
        )
        manager.publish_component("decoder", version, temporary_output, {
            "version": version,
            "fixed_encoder_version": version,
            "previous_decoder_version": version - 1,
            "replay_snapshot": str(snapshot),
            "train_steps": settings["decoder_train_steps"],
        })
        manager.publish_policy(version)
        temporary_output.unlink(missing_ok=True)


def _collector_worker(settings: dict) -> None:
    from scripts.components.collect_data_fm import run_async_collector

    run_async_collector(
        pipeline_root=settings["versions_dir"],
        replay_buffer_dir=settings["replay_dir"],
        stop_file=settings["stop_file"],
        poll_seconds=settings["poll_seconds"],
        rollout_steps=settings["collector_rollout_steps"],
        replay_capacity=settings["replay_capacity"],
        metrics_file=settings["metrics_file"],
    )


def _bootstrap_async_pipeline(
    manager: VersionManager,
    initial_checkpoint: Path,
) -> None:
    """Publish immutable encoder_0 + decoder_0 extracted from one offline checkpoint."""
    with initial_checkpoint.open("rb") as file:
        checkpoint = pickle.load(file)
    bootstrap_dir = manager.root / ".bootstrap"
    bootstrap_dir.mkdir(exist_ok=True)
    encoder_source = atomic_pickle_dump(checkpoint, bootstrap_dir / "encoder_0.pkl")
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
    num_versions: int = 48,
    encoder_train_env_steps: int = 393216,
    decoder_train_steps: int = 10000,
    encoder_demo_ratio: float = 0.5,
    encoder_replay_ratio: float = 0.5,
    minimum_replay_size: int = 5000,
    replay_capacity: int | None = 500000,
    collector_rollout_steps: int = 8,
    poll_seconds: float = 2.0,
) -> None:
    """Run Collector, Encoder Trainer and Decoder Trainer as independent processes."""
    if not np.isclose(encoder_demo_ratio + encoder_replay_ratio, 1.0):
        raise ValueError("encoder demo/replay ratios must sum to 1.0")
    if num_versions < 1:
        raise ValueError("num_versions must be at least 1")
    if minimum_replay_size < 1:
        raise ValueError("minimum_replay_size must be positive")
    config = TrainingConfig().to_dict() | EnvConfig().to_dict()
    from envs.robomimic.RobomimicEnv import RobomimicEnv
    env = RobomimicEnv(dataset_path=config["dataset_path"], reward_shaping=config["dense_reward"])
    obs_dim = int(env.observation_size)
    action_dim = int(env.action_size)
    checkpoint = _validate_offline_checkpoint(
        Path(offline_checkpoint_path), obs_dim, action_dim,
        config["env_name"], require_rlpd_state=True,
    )
    del env
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
    stop = root / "STOP"
    stop.unlink(missing_ok=True)
    manager = VersionManager(versions)
    _bootstrap_async_pipeline(manager, checkpoint)
    settings = {
        "versions_dir": str(versions), "replay_dir": str(replay_dir),
        "snapshots_dir": str(snapshots), "work_dir": str(work),
        "stop_file": str(stop), "demo_buffer_path": str(canonical_demo),
        "metrics_file": str(root / "async_metrics.jsonl"),
        "start_version": 1, "max_version": num_versions,
        "encoder_train_env_steps": encoder_train_env_steps,
        "decoder_train_steps": decoder_train_steps,
        "encoder_demo_ratio": encoder_demo_ratio,
        "encoder_replay_ratio": encoder_replay_ratio,
        "minimum_replay_size": minimum_replay_size,
        "replay_capacity": replay_capacity, "collector_rollout_steps": collector_rollout_steps,
        "poll_seconds": poll_seconds,
    }
    atomic_pickle_dump(settings, root / "pipeline_settings.pkl")
    context = mp.get_context("spawn")
    processes = [
        context.Process(name="data-collector", target=_collector_worker, args=(settings,)),
        context.Process(name="encoder-trainer", target=_encoder_worker, args=(settings,)),
        context.Process(name="decoder-trainer", target=_decoder_worker, args=(settings,)),
    ]
    for process in processes:
        process.start()
    try:
        while any(process.is_alive() for process in processes):
            failed = next((p for p in processes if p.exitcode not in (None, 0)), None)
            if failed is not None:
                raise RuntimeError(f"{failed.name} exited with code {failed.exitcode}")
            # Trainers finish after num_versions; collector is intentionally endless.
            if all(not p.is_alive() for p in processes[1:]):
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping asynchronous pipeline...")
    finally:
        stop.write_text("stop\n", encoding="utf-8")
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
    failures = {
        p.name: p.exitcode for p in processes[1:] if p.exitcode not in (0, None)
    }
    if failures:
        raise RuntimeError(f"Asynchronous pipeline failed: {failures}")
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
                metrics["video/evaluation"] = wandb.Video(video_path)
            wandb_run.log(metrics)


def run_command(
    cmd: str,
    description: str,
    metrics_file: Path | None = None,
    wandb_run=None,
) -> int:
    """Run a command, continuously forwarding child metrics to the parent run."""
    process = subprocess.Popen(cmd, shell=True)
    metrics_offset = 0
    while process.poll() is None:
        if metrics_file is not None and wandb_run is not None:
            metrics_offset = _forward_metrics(metrics_file, wandb_run, metrics_offset)
        time.sleep(0.5)

    if metrics_file is not None and wandb_run is not None:
        _forward_metrics(metrics_file, wandb_run, metrics_offset)

    if process.returncode != 0:
        print(f"ERROR: {description} failed")
        if wandb_run is not None:
            wandb_run.log({"pipeline/failed": 1, "pipeline/failed_step": description})
            wandb_run.finish(exit_code=process.returncode)
        sys.exit(process.returncode)

    return process.returncode


def _tyro_bool_flag(name: str, enabled: bool) -> str:
    """Return the Tyro flag for a boolean option without a separate value."""
    option = name.replace("_", "-")
    return f"--{option}" if enabled else f"--no-{option}"


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
        }
        missing_rlpd_keys = sorted(required_rlpd_keys.difference(checkpoint))
        if missing_rlpd_keys:
            raise ValueError(
                "Offline checkpoint cannot initialize the full RLPD encoder; "
                f"missing keys: {missing_rlpd_keys}."
            )
    return checkpoint_path


def main(
    use_offline_checkpoint: bool = False,
    offline_checkpoint_path: str | None = None,
    stage_init_before_training: bool = False,
    mode: str = "async",
    demo_buffer_path: str | None = None,
    async_run_dir: str | None = None,
    async_num_versions: int = 48,
    encoder_train_env_steps: int = 393216,
    decoder_train_steps: int = 10000,
    encoder_demo_ratio: float = 0.5,
    encoder_replay_ratio: float = 0.5,
    minimum_replay_size: int = 5000,
    replay_capacity: int | None = 500000,
    collector_rollout_steps: int = 8,
    poll_seconds: float = 2.0,
) -> None:
    """Run the complete RLPD encoder + FM decoder training pipeline.

    Args:
        env_name: Environment to train on
        num_stages: Total number of stages (0, 1, 2, ...)
        encoder_num_timesteps: Default training steps for all stages
        encoder_timesteps_per_stage: Comma-separated timesteps per stage
        seed: Random seed
        use_offline_checkpoint: When stage_init_before_training=False, start
            stage 0 from an offline-trained decoder and full RLPD encoder state.
        offline_checkpoint_path: Offline checkpoint containing decoder fields
            and the full ``rlpd_z_*`` actor/Q/observation-stat state.
        stage_init_before_training: Reinitialize encoder and decoder at every
            stage. When enabled, offline checkpoint settings are ignored.
    """
    if mode not in {"legacy", "async"}:
        raise ValueError("mode must be either 'legacy' or 'async'")
    if mode == "async":
        if offline_checkpoint_path is None:
            raise ValueError("async mode requires --offline-checkpoint-path")
        if demo_buffer_path is None:
            raise ValueError("async mode requires --demo-buffer-path")
        run_async_pipeline(
            offline_checkpoint_path=offline_checkpoint_path,
            demo_buffer_path=demo_buffer_path,
            run_dir=async_run_dir,
            num_versions=async_num_versions,
            encoder_train_env_steps=encoder_train_env_steps,
            decoder_train_steps=decoder_train_steps,
            encoder_demo_ratio=encoder_demo_ratio,
            encoder_replay_ratio=encoder_replay_ratio,
            minimum_replay_size=minimum_replay_size,
            replay_capacity=replay_capacity,
            collector_rollout_steps=collector_rollout_steps,
            poll_seconds=poll_seconds,
        )
        return

    config = (
        TrainingConfig().to_dict()
        | EnvConfig().to_dict()
    )
    if (
        not stage_init_before_training
        and use_offline_checkpoint
        and offline_checkpoint_path is None
    ):
        raise ValueError(
            "--offline-checkpoint-path is required when "
            "--use-offline-checkpoint is enabled and "
            "--no-stage-init-before-training is selected."
        )
    offline_checkpoint_requested = (
        not stage_init_before_training
        and use_offline_checkpoint
        and offline_checkpoint_path is not None
    )
    if offline_checkpoint_path is not None and not use_offline_checkpoint:
        print("use_offline_checkpoint=False: ignoring offline_checkpoint_path.")
    if stage_init_before_training and (
        use_offline_checkpoint or offline_checkpoint_path is not None
    ):
        print(
            "stage_init_before_training=True: ignoring offline checkpoint "
            "settings and reinitializing encoder/decoder at every stage."
        )
    # Create unique run identifier
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"rlpd_fm_{config['env_name']}_seed{config['seed']}_{timestamp}"

    # Parse encoder timesteps per stage
    if config['encoder_timesteps_per_stage'] is not None:
        timesteps_list = [int(x.strip()) for x in config['encoder_timesteps_per_stage'].split(",")]
        if len(timesteps_list) < config['num_stages']:
            timesteps_list.extend([config['encoder_num_timesteps']] * (config['num_stages'] - len(timesteps_list)))
        timesteps_list = timesteps_list[:config['num_stages']]
    else:
        timesteps_list = [config['encoder_num_timesteps']] * config['num_stages']

    # Create run directory
    run_dir = Path("results") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # The pipeline parent is the only process that owns a W&B object. Training
    # subprocesses stream JSONL metrics back to this process.
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
                "pipeline_run_id": run_id,
                "use_offline_checkpoint": use_offline_checkpoint,
                "offline_checkpoint_requested": offline_checkpoint_requested,
                "offline_checkpoint_path": offline_checkpoint_path,
            },
        )
        wandb_run.define_metric("pipeline/env_step")
        wandb_run.define_metric("train/*", step_metric="pipeline/env_step")
        wandb_run.define_metric("eval/*", step_metric="pipeline/env_step")
        wandb_run.define_metric("video/*", step_metric="pipeline/env_step")
        wandb_run.define_metric("pipeline/decoder_step")
        wandb_run.define_metric("decoder/*", step_metric="pipeline/decoder_step")

    print(f"\nRLPD + FM - {config['env_name']}")
    print(f"Stages: {config['num_stages']}, Timesteps: {timesteps_list}")

    # Auto-detect action_dim (z_dim) from environment
    # Delay the environment import so CLI help does not require all simulation
    # and dataset runtime dependencies to be installed and importable.
    from envs.robomimic.RobomimicEnv import RobomimicEnv

    env = RobomimicEnv(dataset_path=config['dataset_path'], reward_shaping=config['dense_reward'])
    z_dim = env.action_size
    obs_dim = env.observation_size
    del env
    initial_offline_checkpoint = None
    if offline_checkpoint_requested:
        initial_offline_checkpoint = _validate_offline_checkpoint(
            Path(offline_checkpoint_path),
            expected_obs_dim=obs_dim,
            expected_action_dim=z_dim,
            expected_env_name=config["env_name"],
            require_rlpd_state=True,
        )
        print(
            "Initial offline decoder + RLPD encoder checkpoint: "
            f"{initial_offline_checkpoint}"
        )

    # Save pipeline config
    config_file = run_dir / "pipeline_config.txt"
    with open(config_file, "w") as f:
        f.write(f"RLPD + FM Configuration\n")
        f.write(f"{'='*60}\n")
        f.write(f"Environment: {config['env_name']}\n")
        f.write(f"Stages: {config['num_stages']}\n")
        f.write(f"Timesteps per stage: {timesteps_list}\n")
        f.write(f"Seed: {config['seed']}\n")
        f.write(f"z_dim: {z_dim}\n")
        f.write(f"Use offline checkpoint: {use_offline_checkpoint}\n")
        f.write(f"Offline checkpoint requested: {offline_checkpoint_requested}\n")
        f.write(f"Offline checkpoint path: {offline_checkpoint_path}\n")
        f.write(f"Initial offline checkpoint: {initial_offline_checkpoint}\n")

    # Create global continuous metrics file
    global_metrics_file = run_dir / "global_eval_metrics.txt"
    with open(global_metrics_file, "w") as f:
        f.write(f"RLPD + FM Evaluation Metrics\n")
        f.write(f"Environment: {config['env_name']}\n")
        f.write(f"Stages: {config['num_stages']}\n\n")

    # Track checkpoints and cumulative steps across stages
    fm_checkpoint = None
    encoder_checkpoint = None
    cumulative_step = 0
    encoder_replay_path = run_dir / "encoder_replay_buffer.pkl"
    decoder_replay_path = run_dir / "decoder_replay_buffer.pkl"

    for stage in range(config['num_stages']):
        print(f"\n=== Stage {stage}/{config['num_stages']-1} ===")

        stage_dir = run_dir / f"stage_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        encoder_metrics_file = stage_dir / "encoder_wandb_metrics.jsonl"
        decoder_metrics_file = stage_dir / "decoder_wandb_metrics.jsonl"
        if wandb_run is not None:
            wandb_run.log({
                "pipeline/stage": stage,
                "stage/started": 1,
                f"stage_{stage}/started": 1,
            })

        # =====================================================================
        # STEP 1: Init decoder (only for stage 0)
        # =====================================================================
        if stage == 0 and initial_offline_checkpoint is not None:
            fm_checkpoint = initial_offline_checkpoint
            encoder_checkpoint = initial_offline_checkpoint
            print(
                "Using offline checkpoint as Encoder_0 and Decoder_0: "
                f"{initial_offline_checkpoint}"
            )
            if wandb_run is not None:
                wandb_run.log({
                    "pipeline/stage": stage,
                    "pipeline/phase": "load_offline_decoder",
                })
        elif stage == 0:
            if wandb_run is not None:
                wandb_run.log({"pipeline/stage": stage, "pipeline/phase": "init_decoder"})
            cmd = (
                f"python scripts/components/init_decoder_fm.py "
                f"--env_name {config['env_name']} "
                f"--dataset_path {config['dataset_path']} "
                f"--output_dir {stage_dir} "
                f"--seed {config['seed']}"
            )
            run_command(cmd, f"Stage {stage}: Init decoder", wandb_run=wandb_run)

            fm_files = list(stage_dir.glob(f"fm_identity_{config['env_name']}_*.pkl"))
            if not fm_files:
                print(f"ERROR: Identity decoder not found in {stage_dir}")
                sys.exit(1)
            fm_checkpoint = sorted(fm_files)[-1]

        # =====================================================================
        # STEP 2: Encoder update
        # =====================================================================
        if fm_checkpoint is None:
            print(f"ERROR: No decoder checkpoint available for stage {stage}")
            sys.exit(1)

        encoder_exp_name = f"pipeline_{run_id}_stage{stage}"
        stage_timesteps = timesteps_list[stage]
        stage_step_offset = sum(timesteps_list[:stage])

        encoder_source_checkpoint = encoder_checkpoint
        cmd_parts = [
            "MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python scripts/components/train_encoder_rlpd.py",
            f"--decoder_model_path {shlex.quote(str(fm_checkpoint))}",
            f"--exp_name {shlex.quote(encoder_exp_name)}",
            f"--num_timesteps {stage_timesteps}",
            f"--stage {stage}",
            f"--global_step_offset {stage_step_offset}",
            f"--metrics_file {shlex.quote(str(encoder_metrics_file))}",
            f"--replay_buffer_path {shlex.quote(str(encoder_replay_path))}",
            _tyro_bool_flag(
                "stage_init_before_training", stage_init_before_training
            ),
        ]
        if not stage_init_before_training and encoder_source_checkpoint is not None:
            cmd_parts.append(
                "--encoder_model_path "
                f"{shlex.quote(str(encoder_source_checkpoint))}"
            )
        cmd = " ".join(cmd_parts)

        if wandb_run is not None:
            wandb_run.log({"pipeline/stage": stage, "pipeline/phase": "train_encoder"})
        run_command(
            cmd,
            f"Stage {stage}: Encoder update",
            encoder_metrics_file,
            wandb_run,
        )

        # Find encoder checkpoint
        encoder_pattern = f"encoder_rlpd_fm_{config['env_name']}_{encoder_exp_name}_*"
        encoder_results = list(Path("results").glob(encoder_pattern))

        if not encoder_results:
            print(f"ERROR: Encoder results not found")
            sys.exit(1)

        encoder_result_dir = sorted(encoder_results)[-1]
        encoder_checkpoint = encoder_result_dir / "best_checkpoint.pkl"

        if not encoder_checkpoint.exists():
            print(f"ERROR: Encoder checkpoint not found: {encoder_checkpoint}")
            sys.exit(1)

        # Merge evaluation metrics to global file
        eval_metrics_file = encoder_result_dir / "eval_metrics.txt"
        if eval_metrics_file.exists():
            with open(eval_metrics_file, "r") as f_in:
                lines = f_in.readlines()

            max_local_step = 0
            with open(global_metrics_file, "a") as f_out:
                f_out.write(f"\nSTAGE {stage}\n")
                for line in lines:
                    if line.startswith("Step:"):
                        local_step = int(line.split(":")[1].strip())
                        global_step = cumulative_step + local_step
                        max_local_step = max(max_local_step, local_step)
                        f_out.write(f"Step: {global_step}\n")
                    else:
                        f_out.write(line)

            cumulative_step += (max_local_step + 1)

        # =====================================================================
        # STEP 3: Collect data
        # =====================================================================
        cmd = (
            f"python scripts/components/collect_data_fm.py "
            f"--ppo_z_checkpoint_path {shlex.quote(str(encoder_checkpoint))} "
            f"--fm_model_path {shlex.quote(str(fm_checkpoint))} "
            f"--output_dir {shlex.quote(str(stage_dir))} "
        )
        if wandb_run is not None:
            wandb_run.log({"pipeline/stage": stage, "pipeline/phase": "collect_data"})
        run_command(cmd, f"Stage {stage}: Collect data", wandb_run=wandb_run)

        data_files = list(stage_dir.glob(f"rlpd_z_fm_data_{config['env_name']}_*.pkl"))
        if not data_files:
            print(f"ERROR: Data file not found in {stage_dir}")
            sys.exit(1)
        data_file = sorted(data_files)[-1]

        # Decoder replay persists across stages. It is only truncated when its
        # configured capacity is reached; a partially filled buffer is never reset.
        with open(data_file, "rb") as f:
            stage_data = pickle.load(f)
        if decoder_replay_path.is_file():
            with open(decoder_replay_path, "rb") as f:
                decoder_data = pickle.load(f)
            for key in ("states", "actions", "rewards"):
                combined = np.concatenate(
                    [decoder_data[key], stage_data[key]], axis=0
                )
                capacity = config["fm_max_samples"]
                decoder_data[key] = (
                    combined if capacity is None else combined[-capacity:]
                )
            decoder_data.update(
                {key: value for key, value in stage_data.items() if key not in ("states", "actions", "rewards")}
            )
        else:
            decoder_data = stage_data
        with open(decoder_replay_path, "wb") as f:
            pickle.dump(decoder_data, f)

        # =====================================================================
        # STEP 4: Decoder update
        # =====================================================================
        fm_cmd_parts = [
            "python scripts/components/train_decoder_fm.py",
            f"--data_path {shlex.quote(str(decoder_replay_path))}",
            f"--output_dir {shlex.quote(str(stage_dir))}",
            f"--stage {stage}",
            f"--global_epoch_offset {stage * config['fm_num_epochs']}",
            f"--metrics_file {decoder_metrics_file}",
            _tyro_bool_flag(
                "stage_init_before_training", stage_init_before_training
            ),
        ]

        cmd = " ".join(fm_cmd_parts)
        if wandb_run is not None:
            wandb_run.log({"pipeline/stage": stage, "pipeline/phase": "train_decoder"})
        run_command(
            cmd,
            f"Stage {stage}: Decoder update",
            decoder_metrics_file,
            wandb_run,
        )

        fm_best_files = list(stage_dir.glob(f"fm_model_best_*.pkl"))
        if not fm_best_files:
            print(f"ERROR: Decoder checkpoint not found in {stage_dir}")
            sys.exit(1)
        fm_checkpoint = sorted(fm_best_files)[-1]

        # Save stage summary
        summary_file = stage_dir / "stage_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Stage {stage} Summary\n")
            f.write(f"Decoder: {fm_checkpoint}\n")
            f.write(f"Encoder: {encoder_checkpoint}\n")
            f.write(f"Data: {data_file}\n")

    # =========================================================================
    # Pipeline Complete
    # =========================================================================
    print(f"\nRLPD + FM complete - {config['num_stages']} stages")
    print(f"Output: {run_dir}")

    # Save final summary
    final_summary = run_dir / "final_summary.txt"
    with open(final_summary, "w") as f:
        f.write(f"RLPD + FM Summary\n")
        f.write(f"Environment: {config['env_name']}\n")
        f.write(f"Stages: {config['num_stages']}\n")
        f.write(f"Final decoder: {fm_checkpoint}\n")
        f.write(f"Final encoder: {encoder_checkpoint}\n")

    if wandb_run is not None:
        wandb_run.log({
            "pipeline/completed": 1,
            "pipeline/completed_stages": config["num_stages"],
        })
        wandb_run.finish()


if __name__ == "__main__":
    tyro.cli(main)
