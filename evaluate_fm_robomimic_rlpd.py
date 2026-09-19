"""Evaluate offline-frozen and online GoRL FM checkpoints on Robomimic.

Offline ``checkpoint_final.pkl`` files are self-contained and contain an IQL
actor used to warm-start RLPD. Online RLPD runs may write a self-contained
encoder checkpoint or separate encoder and decoder checkpoints, so pass one as
``--checkpoint`` and the other through ``--decoder-checkpoint`` or
``--encoder-checkpoint``. Evaluation executes encoder -> z -> decoder ->
Robomimic action without applying an extra tanh to the decoder output.
"""

from __future__ import annotations

import datetime
import json
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import jax
import jax_dataclasses as jdc
import cv2
import numpy as np
import tyro
from jax import numpy as jnp

from envs.robomimic.online_config.env_config import EnvConfig
# Importing ``robomimic.utils.file_utils`` also imports its language utilities,
# which eagerly construct CLIP. Evaluation only needs dataset metadata, and a
# missing internet route must not make startup hang while Hugging Face checks
# for updates. Cached files remain available in offline mode.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = str(3)
@dataclass
class EvaluationConfig:
    """CLI configuration for Robomimic policy evaluation."""

    checkpoint: str
    encoder_checkpoint: str | None = None
    decoder_checkpoint: str | None = None
    dataset_path: str | None = None
    episodes: int = 20
    episode_length: int | None = None
    seed: int = 0
    deterministic: bool = True
    apply_tanh: bool | None = None
    render: bool = False
    render_camera: str = "agentview"
    video_dir: str | None = "evaluation_videos/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_fps: int = 20
    video_skip: int = 1
    video_height: int = 512
    video_width: int = 512
    output_json: str | None = None


@dataclass(frozen=True)
class LoadedPolicy:
    checkpoint_path: Path
    encoder_checkpoint_path: Path
    decoder_checkpoint_path: Path
    env: Any
    decoder: Any
    actor_params: Any
    actor_obs_stats: Any
    normalize_actor_observations: bool
    apply_tanh: bool
    encoder_algorithm: str
    decoder_type: str
    episode_length: int
    checkpoint_kind: str


def _make_evaluation_env(
    dataset_path: Path,
    reward_shaping: bool,
    render_offscreen: bool,
) -> Any:
    """Build a Robomimic env whose flat observations match dataset order.

    ``RobomimicEnv.flatten_obs_dict`` follows the runtime dictionary insertion
    order. Robosuite's runtime order is not guaranteed to match the HDF5
    observation-key order used to train the checkpoints, even when both flatten
    to the same total size. Keep this evaluation-only compatibility behavior in
    this file instead of changing the shared environment implementation.
    """
    from envs.robomimic.RobomimicEnv import RobomimicEnv

    class DatasetOrderedRobomimicEnv(RobomimicEnv):
        def flatten_obs_dict(self, obs_dict: Mapping[str, Any]) -> jax.Array:
            missing = [key for key in self.obs_keys if key not in obs_dict]
            if missing:
                raise KeyError(
                    "Robomimic environment observation is missing dataset keys: "
                    + ", ".join(missing)
                )

            values = []
            for key in self.obs_keys:
                value = jnp.ravel(jnp.asarray(obs_dict[key]))
                expected_shape = tuple(self.shape_meta["all_shapes"][key])
                expected_size = int(np.prod(expected_shape))
                if value.size != expected_size:
                    raise ValueError(
                        f"Observation key {key!r} has {value.size} values at "
                        f"runtime, but dataset shape {expected_shape} requires "
                        f"{expected_size}."
                    )
                values.append(value)
            return jnp.concatenate(values, axis=0)

    return DatasetOrderedRobomimicEnv(
        dataset_path=str(dataset_path),
        render_offscreen=render_offscreen,
        reward_shaping=reward_shaping,
    )


def _load_checkpoint(path: str) -> tuple[Path, Mapping[str, Any]]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    with checkpoint_path.open("rb") as file:
        checkpoint = pickle.load(file)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Expected a checkpoint mapping, got {type(checkpoint).__name__}."
        )
    return checkpoint_path, checkpoint


def _config_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        return asdict(value)
    except (TypeError, ValueError):
        return vars(value) if hasattr(value, "__dict__") else {}


def _resolve_dataset_path(
    checkpoints: tuple[Mapping[str, Any], ...], config: EvaluationConfig
) -> Path:
    candidate: Any = config.dataset_path
    for checkpoint in checkpoints:
        offline_config = _config_dict(checkpoint.get("offline_config", {}))
        candidate = (
            candidate
            or checkpoint.get("dataset_path")
            or offline_config.get("dataset_path")
        )
    if candidate is None:
        # Online encoder / decoder checkpoints predate embedded environment
        # metadata. Use the same project-owned config as run_gorl_fm.py.
        from envs.robomimic.online_config.env_config import EnvConfig

        candidate = EnvConfig().dataset_path
    if candidate is None:
        raise ValueError(
            "Could not infer the Robomimic dataset path from the checkpoint. "
            "Pass --dataset-path explicitly."
        )
    path = Path(str(candidate)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Robomimic dataset does not exist: {path}")
    return path


def _decoder_fields(
    parameter_checkpoint: Mapping[str, Any],
    config_checkpoint: Mapping[str, Any],
    env: Any,
) -> tuple[Any, Any, Any, int, int, str]:
    from flow_policy.decoder_1step_fm_residualMLP import Decoder1StepFMConfig
    from flow_policy.decoder_fm import DecoderFMConfig

    params = parameter_checkpoint.get(
        "fm_params", parameter_checkpoint.get("params")
    )
    obs_stats = parameter_checkpoint.get(
        "fm_obs_stats", parameter_checkpoint.get("obs_stats")
    )
    decoder_config = config_checkpoint.get("config")
    obs_dim = config_checkpoint.get(
        "obs_dim", parameter_checkpoint.get("obs_dim", env.observation_size)
    )
    action_dim = config_checkpoint.get(
        "action_dim",
        parameter_checkpoint.get(
            "action_dim", parameter_checkpoint.get("z_dim", env.action_size)
        ),
    )
    missing = [
        name
        for name, value in (
            ("params/fm_params", params),
            ("obs_stats/fm_obs_stats", obs_stats),
            ("config", decoder_config),
        )
        if value is None
    ]
    if missing:
        raise KeyError("Checkpoint is missing decoder fields: " + ", ".join(missing))

    if isinstance(decoder_config, Decoder1StepFMConfig):
        config_decoder_type = "meanflow"
    elif isinstance(decoder_config, DecoderFMConfig):
        config_decoder_type = "flow_matching"
    else:
        raise TypeError(
            "Decoder checkpoint 'config' must be Decoder1StepFMConfig "
            "(meanflow) or DecoderFMConfig (flow_matching), got "
            f"{type(decoder_config).__name__}."
        )
    checkpoint_decoder_type = config_checkpoint.get("decoder_type")
    if checkpoint_decoder_type is None:
        checkpoint_decoder_type = config_decoder_type
    if checkpoint_decoder_type == "fm":
        checkpoint_decoder_type = "flow_matching"
    if checkpoint_decoder_type not in ("meanflow", "flow_matching"):
        raise ValueError(
            "Unsupported decoder_type in checkpoint: "
            f"{checkpoint_decoder_type!r}. Expected 'meanflow' or 'flow_matching'."
        )
    if checkpoint_decoder_type != config_decoder_type:
        raise ValueError(
            "Decoder checkpoint decoder_type does not match its config: "
            f"decoder_type={checkpoint_decoder_type!r}, "
            f"config={config_decoder_type!r}."
        )
    return (
        params,
        obs_stats,
        decoder_config,
        int(obs_dim),
        int(action_dim),
        checkpoint_decoder_type,
    )


def _actor_params(checkpoint: Mapping[str, Any]) -> Any:
    for key in ("rlpd_z_actor_params", "iql_z_actor_params"):
        if key in checkpoint:
            return checkpoint[key]
    raise KeyError(
        "Checkpoint has neither 'rlpd_z_actor_params' nor "
        "'iql_z_actor_params'. Pass a checkpoint produced by "
        "scripts/run_rlpd_fm.py or run_offline_fm_frozen_robomimic.py."
    )


def _actor_obs_stats(checkpoint: Mapping[str, Any]) -> Any:
    for key in ("rlpd_z_obs_stats", "iql_z_obs_stats"):
        if key in checkpoint:
            return checkpoint[key]
    raise KeyError(
        "Checkpoint has neither 'rlpd_z_obs_stats' nor 'iql_z_obs_stats'."
    )


def _encoder_algorithm(checkpoint: Mapping[str, Any]) -> str:
    if "rlpd_z_actor_params" in checkpoint:
        return "rlpd"
    if "iql_z_actor_params" in checkpoint:
        return "offline-iql-warm-start"
    return "unknown"


def _checkpoint_kind(checkpoint: Mapping[str, Any]) -> str:
    if checkpoint.get("is_frozen_offline"):
        return "offline-frozen"
    if str(checkpoint.get("checkpoint_format", "")).startswith("gorl_offline"):
        return "offline-frozen"
    if "rlpd_z_actor_params" in checkpoint:
        return "gorl-online-rlpd"
    return "gorl-online"


def _has_encoder(checkpoint: Mapping[str, Any]) -> bool:
    return any(
        all(key in checkpoint for key in keys)
        for keys in (
            ("rlpd_z_actor_params", "rlpd_z_obs_stats"),
            ("iql_z_actor_params", "iql_z_obs_stats"),
        )
    )


def _has_decoder_config(checkpoint: Mapping[str, Any]) -> bool:
    from flow_policy.decoder_1step_fm_residualMLP import Decoder1StepFMConfig
    from flow_policy.decoder_fm import DecoderFMConfig

    return isinstance(
        checkpoint.get("config"), (Decoder1StepFMConfig, DecoderFMConfig)
    )


def _select_checkpoint_roles(
    config: EvaluationConfig,
) -> tuple[
    Path,
    Mapping[str, Any],
    Path,
    Mapping[str, Any],
    Path,
    Mapping[str, Any],
]:
    primary_path, primary = _load_checkpoint(config.checkpoint)
    encoder_path, encoder = primary_path, primary
    decoder_path, decoder = primary_path, primary
    if config.encoder_checkpoint is not None:
        encoder_path, encoder = _load_checkpoint(config.encoder_checkpoint)
    if config.decoder_checkpoint is not None:
        decoder_path, decoder = _load_checkpoint(config.decoder_checkpoint)

    if not _has_encoder(encoder):
        if _has_encoder(primary):
            encoder_path, encoder = primary_path, primary
        else:
            raise KeyError(
                "No encoder fields were found. If --checkpoint is a GoRL "
                "fm_model_*.pkl, pass the matching best_checkpoint.pkl or "
                "final_checkpoint.pkl through --encoder-checkpoint."
            )
    if not _has_decoder_config(decoder):
        if _has_decoder_config(primary):
            decoder_path, decoder = primary_path, primary
        else:
            raise TypeError(
                "No supported decoder config was found. If --checkpoint is a GoRL "
                "encoder best/final_checkpoint.pkl, pass the matching stage "
                "fm_model_*.pkl through --decoder-checkpoint."
            )
    return primary_path, primary, encoder_path, encoder, decoder_path, decoder


def load_policy(config: EvaluationConfig) -> LoadedPolicy:
    from flow_policy import networks
    from flow_policy.decoder_1step_fm_residualMLP import Decoder1StepFMState
    from flow_policy.decoder_fm import DecoderFMState
    from flow_policy.config_utils import fill_unspecified_config_values

    (
        checkpoint_path,
        checkpoint,
        encoder_path,
        encoder_checkpoint,
        decoder_path,
        decoder_checkpoint,
    ) = _select_checkpoint_roles(config)
    dataset_path = _resolve_dataset_path(
        (checkpoint, encoder_checkpoint, decoder_checkpoint), config
    )
    env_config = EnvConfig().to_dict()
    env = _make_evaluation_env(
        dataset_path,
        env_config['dense_reward'],
        render_offscreen=config.video_dir is not None,
    )
    # An online encoder checkpoint embeds the decoder that was used to train
    # that encoder. A separately supplied stage decoder is newer and must take
    # precedence when the caller explicitly requests it.
    decoder_parameter_checkpoint = (
        decoder_checkpoint
        if config.decoder_checkpoint is not None
        else encoder_checkpoint
        if "fm_params" in encoder_checkpoint
        else decoder_checkpoint
    )
    (
        params,
        obs_stats,
        decoder_config,
        obs_dim,
        action_dim,
        decoder_type,
    ) = _decoder_fields(
        decoder_parameter_checkpoint, decoder_checkpoint, env
    )
    if (obs_dim, action_dim) != (env.observation_size, env.action_size):
        raise ValueError(
            "Checkpoint dimensions do not match the Robomimic dataset: "
            f"checkpoint=({obs_dim}, {action_dim}), "
            f"environment=({env.observation_size}, {env.action_size})."
        )

    decoder_state_cls = (
        Decoder1StepFMState if decoder_type == "meanflow" else DecoderFMState
    )
    if decoder_type == "meanflow":
        from envs.robomimic.online_config.decoder_configs.meanflow_config import (
            MeanFlowConfig,
        )

        decoder_config = fill_unspecified_config_values(
            decoder_config,
            warm_up_epoch=MeanFlowConfig().meanflow_warm_up_epoch,
        )
    decoder = decoder_state_cls.init(
        jax.random.key(config.seed + 1), obs_dim, action_dim, decoder_config
    )
    with jdc.copy_and_mutate(decoder) as decoder:
        decoder.params = params
        decoder.obs_stats = obs_stats

    actor_obs_stats = _actor_obs_stats(encoder_checkpoint)

    encoder_config = encoder_checkpoint.get(
        "rlpd_encoder_config",
        encoder_checkpoint.get(
            "online_encoder_config", encoder_checkpoint.get("config")
        ),
    )
    normalize_actor_observations = bool(
        getattr(encoder_config, "normalize_observations", True)
    )
    # Current RLPD rollouts pass the frozen decoder output directly to the
    # environment. ``apply_tanh_in_rollout`` remains in training configs for
    # compatibility but rollout_encoder.py intentionally does not use it.
    default_apply_tanh = False
    apply_tanh = (
        default_apply_tanh if config.apply_tanh is None else config.apply_tanh
    )
    offline_config = _config_dict(
        encoder_checkpoint.get("offline_config", {})
    )
    configured_episode_length = getattr(encoder_config, "episode_length", None)
    episode_length = int(
        config.episode_length
        if config.episode_length is not None
        else offline_config.get(
            "episode_length",
            configured_episode_length
            if configured_episode_length is not None
            else 150,
        )
    )
    if config.episodes < 1 or episode_length < 1:
        raise ValueError("episodes and episode_length must be positive.")

    actor_params = _actor_params(encoder_checkpoint)
    dummy_obs = jnp.zeros((obs_dim,))
    if normalize_actor_observations:
        dummy_obs = (dummy_obs - actor_obs_stats.mean) / (
            actor_obs_stats.std + 1e-8
        )
    distribution = networks.gaussian_policy_fwd(
        actor_params, dummy_obs
    )
    if distribution.loc.shape[-1] != action_dim:
        raise ValueError(
            f"Encoder latent dimension {distribution.loc.shape[-1]} does not "
            f"match decoder action dimension {action_dim}."
        )

    return LoadedPolicy(
        checkpoint_path=checkpoint_path,
        encoder_checkpoint_path=encoder_path,
        decoder_checkpoint_path=decoder_path,
        env=env,
        decoder=decoder,
        actor_params=actor_params,
        actor_obs_stats=actor_obs_stats,
        normalize_actor_observations=normalize_actor_observations,
        apply_tanh=apply_tanh,
        encoder_algorithm=_encoder_algorithm(encoder_checkpoint),
        decoder_type=decoder_type,
        episode_length=episode_length,
        checkpoint_kind=_checkpoint_kind(encoder_checkpoint),
    )


def _policy_action(
    policy: LoadedPolicy,
    observation: jax.Array,
    key: jax.Array,
    deterministic: bool,
) -> jax.Array:
    from flow_policy import networks

    actor_observation = observation
    if policy.normalize_actor_observations:
        actor_observation = (
            observation - policy.actor_obs_stats.mean
        ) / (policy.actor_obs_stats.std + 1e-8)
    distribution = networks.gaussian_policy_fwd(
        policy.actor_params,
        actor_observation,
    )
    latent = distribution.loc if deterministic else distribution.sample(key)
    action = policy.decoder.sample_action_from_z(
        observation, latent, key, deterministic=True
    )
    return jnp.tanh(action) if policy.apply_tanh else action


def _success(info: Mapping[str, Any]) -> bool:
    for key in ("success", "task_success", "is_success"):
        if key in info:
            value = info[key]
            if isinstance(value, Mapping):
                value = value.get("task", False)
            return bool(np.asarray(value))
    return False


def _render_video_frame(policy: LoadedPolicy, config: EvaluationConfig) -> np.ndarray:
    frame = policy.env.render(
        mode="rgb_array",
        height=config.video_height,
        width=config.video_width,
        camera_name=config.render_camera,
    )
    frame_array = np.asarray(frame)
    if frame_array.ndim != 3 or frame_array.shape[-1] not in (3, 4):
        raise ValueError(
            "Expected an RGB or RGBA video frame, got shape "
            f"{frame_array.shape}."
        )
    if frame_array.shape[-1] == 4:
        frame_array = frame_array[..., :3]
    return np.ascontiguousarray(frame_array, dtype=np.uint8)


def _open_video_writer(
    output_path: Path, config: EvaluationConfig
) -> cv2.VideoWriter:
    """Create the same MP4V writer used by run_gorl_fm evaluation videos."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        config.video_fps,
        (config.video_width, config.video_height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            f"OpenCV could not initialize the MP4 video writer: {output_path}"
        )
    return writer


def _write_video_frame(
    writer: cv2.VideoWriter,
    policy: LoadedPolicy,
    config: EvaluationConfig,
) -> None:
    frame = _render_video_frame(policy, config)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def evaluate(config: EvaluationConfig) -> dict[str, Any]:
    if config.video_fps < 1:
        raise ValueError("video_fps must be positive.")
    if config.video_skip < 1:
        raise ValueError("video_skip must be positive.")
    if config.video_height < 1 or config.video_width < 1:
        raise ValueError("video_height and video_width must be positive.")
    if config.render and config.video_dir is not None:
        raise ValueError(
            "On-screen rendering and off-screen video recording cannot be "
            "enabled together. Use --no-render when passing --video-dir."
        )

    policy = load_policy(config)
    env_config = EnvConfig().to_dict()
    print(
        f"Loaded {policy.checkpoint_kind} checkpoint: {policy.checkpoint_path}\n"
        f"Encoder checkpoint: {policy.encoder_checkpoint_path}\n"
        f"Decoder checkpoint: {policy.decoder_checkpoint_path}\n"
        f"Robomimic dimensions: obs={policy.env.observation_size}, "
        f"action={policy.env.action_size}; encoder={policy.encoder_algorithm}; "
        f"apply_tanh={policy.apply_tanh}"
    )
    key = jax.random.key(config.seed)
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    video_paths: list[str] = []
    video_dir = (
        Path(config.video_dir).expanduser().resolve()
        if config.video_dir is not None
        else None
    )
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"Recording every episode to: {video_dir}")
    try:
        for episode in range(config.episodes):
            key, reset_key = jax.random.split(key)
            state = policy.env.reset(reset_key)
            episode_return = 0.0
            success = False
            length = 0
            video_writer = None
            episode_video_path = None
            last_recorded_step = -1
            try:
                if video_dir is not None:
                    episode_video_path = video_dir / f"episode_{episode + 1:04d}.mp4"
                    video_writer = _open_video_writer(episode_video_path, config)
                    _write_video_frame(video_writer, policy, config)
                    last_recorded_step = 0

                for step in range(policy.episode_length):
                    key, action_key = jax.random.split(key)
                    action = _policy_action(
                        policy, state.obs, action_key, config.deterministic
                    )
                    state = policy.env.step(state, action)
                    step_success = _success(state.info)
                    step_reward = float(np.asarray(state.reward))
                    if step_success and env_config["dense_reward"]:
                        step_reward += env_config["success_reward_bonus"]
                    episode_return += step_reward
                    length = step + 1
                    success = success or step_success
                    if config.render:
                        policy.env.render(
                            mode="human", camera_name=config.render_camera
                        )
                    if video_writer is not None and length % config.video_skip == 0:
                        _write_video_frame(video_writer, policy, config)
                        last_recorded_step = length
                    if step_success or bool(np.asarray(state.done)):
                        break

                # Match run_gorl_fm's recorder by always preserving the terminal
                # frame, including when frame skipping omitted the final step.
                if video_writer is not None and last_recorded_step != length:
                    _write_video_frame(video_writer, policy, config)
            finally:
                if video_writer is not None:
                    video_writer.release()
                    assert episode_video_path is not None
                    video_paths.append(str(episode_video_path))
            returns.append(episode_return)
            lengths.append(length)
            successes.append(float(success))
            print(
                f"Episode {episode + 1:>3}/{config.episodes}: "
                f"return={episode_return:9.3f}, length={length:4d}, "
                f"success={success}"
                + (
                    f", video={episode_video_path}"
                    if episode_video_path is not None
                    else ""
                )
            )
    finally:
        close = getattr(policy.env.env, "close", None)
        if callable(close):
            close()

    result = {
        "checkpoint": str(policy.checkpoint_path),
        "encoder_checkpoint": str(policy.encoder_checkpoint_path),
        "decoder_checkpoint": str(policy.decoder_checkpoint_path),
        "checkpoint_kind": policy.checkpoint_kind,
        "encoder_algorithm": policy.encoder_algorithm,
        "decoder_type": policy.decoder_type,
        "dataset_path": policy.env.dataset_path,
        "episodes": config.episodes,
        "episode_length_limit": policy.episode_length,
        "deterministic": config.deterministic,
        "apply_tanh": policy.apply_tanh,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "length_mean": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
        "episode_returns": returns,
        "episode_lengths": lengths,
        "episode_successes": successes,
        "video_dir": str(video_dir) if video_dir is not None else None,
        "episode_videos": video_paths,
    }
    print("\n" + json.dumps(result, indent=2))
    if config.output_json is not None:
        output_path = Path(config.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as file:
            json.dump(result, file, indent=2)
        print(f"Saved evaluation results to {output_path}")
    return result


if __name__ == "__main__":
    evaluate(tyro.cli(EvaluationConfig))
