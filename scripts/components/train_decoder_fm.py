"""Train Flow Matching model to learn PPO action distribution."""

import datetime
import pickle
import time
from pathlib import Path
from typing import Any

import jax
import jax_dataclasses as jdc
import numpy as np
import tyro
from jax import numpy as jnp
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from flow_policy.decoder_fm import DecoderFMConfig, DecoderFMState
from flow_policy.decoder_1step_fm_residualMLP import Decoder1StepFMConfig, Decoder1StepFMState
from flow_policy import networks
from envs.robomimic.online_config.training_config import TrainingConfig
from envs.robomimic.online_config.env_config import EnvConfig
try:
    from .metrics_ipc import append_metrics
    from .online_pipeline_ipc import atomic_pickle_dump, load_transition_data
except ImportError:  # Direct execution: python scripts/components/train_decoder_fm.py
    from metrics_ipc import append_metrics
    from online_pipeline_ipc import atomic_pickle_dump, load_transition_data


def evaluate_decoder_update_need(
    encoder_checkpoint_path: str,
    decoder_checkpoint_path: str,
    replay_snapshot_path: str,
    new_sample_count: int,
    sample_count: int,
    action_roundtrip_p95_threshold: float,
    latent_cycle_p95_threshold: float,
    inverse_outlier_fraction_threshold: float,
    latent_support_radius: float,
    decoder_type: str,
    seed: int,
) -> dict[str, float | int | bool]:
    """Measure whether the current decoder still serves new data and policy latents."""
    with Path(encoder_checkpoint_path).expanduser().open("rb") as file:
        encoder = pickle.load(file)
    with Path(decoder_checkpoint_path).expanduser().open("rb") as file:
        decoder_checkpoint = pickle.load(file)
    replay = load_transition_data(replay_snapshot_path)
    available = min(max(1, int(new_sample_count)), len(replay["observations"]))
    recent_start = len(replay["observations"]) - available
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        recent_start,
        len(replay["observations"]),
        size=min(int(sample_count), available),
    )
    observations = jnp.asarray(replay["observations"][indices])
    actions = jnp.asarray(replay["actions"][indices])
    state_cls = Decoder1StepFMState if decoder_type == "meanflow" else DecoderFMState
    decoder = state_cls.init(
        jax.random.PRNGKey(seed + 1),
        int(decoder_checkpoint["obs_dim"]),
        int(decoder_checkpoint["action_dim"]),
        decoder_checkpoint["config"],
    )
    with jdc.copy_and_mutate(decoder) as decoder:
        decoder.params = decoder_checkpoint["params"]
        decoder.obs_stats = decoder_checkpoint["obs_stats"]
    inverse = jax.jit(decoder.inverse_fm_batch)
    decode = jax.jit(
        lambda obs, z: decoder.sample_action_from_z(
            obs, z, jax.random.PRNGKey(0), deterministic=True
        )
    )
    inverse_latents = inverse(observations, actions)
    reconstructed_actions = decode(observations, inverse_latents)
    action_errors = jnp.mean(jnp.square(reconstructed_actions - actions), axis=-1)

    encoder_config = encoder.get("config", encoder.get("rlpd_encoder_config"))
    actor_obs = observations
    if bool(getattr(encoder_config, "normalize_observations", True)):
        stats = encoder["rlpd_z_obs_stats"]
        actor_obs = (actor_obs - stats.mean) / (stats.std + 1e-8)
    actor_distribution = networks.gaussian_policy_fwd(
        encoder["rlpd_z_actor_params"],
        actor_obs,
        mean_bound=getattr(encoder_config, "actor_mean_bound", None),
    )
    policy_latents = actor_distribution.sample(jax.random.PRNGKey(seed + 2))
    policy_actions = decode(observations, policy_latents)
    roundtrip_policy_latents = inverse(observations, policy_actions)
    latent_errors = jnp.mean(
        jnp.square(roundtrip_policy_latents - policy_latents), axis=-1
    )
    outliers = jnp.any(jnp.abs(inverse_latents) > latent_support_radius, axis=-1)
    action_p95 = float(np.asarray(jnp.percentile(action_errors, 95.0)))
    latent_p95 = float(np.asarray(jnp.percentile(latent_errors, 95.0)))
    outlier_fraction = float(np.asarray(jnp.mean(outliers)))
    action_triggered = action_p95 > action_roundtrip_p95_threshold
    latent_triggered = latent_p95 > latent_cycle_p95_threshold
    outlier_triggered = outlier_fraction > inverse_outlier_fraction_threshold
    return {
        "update_required": bool(
            action_triggered or latent_triggered or outlier_triggered
        ),
        "sample_count": len(indices),
        "new_sample_count": int(new_sample_count),
        "action_roundtrip_p95": action_p95,
        "action_roundtrip_p95_threshold": action_roundtrip_p95_threshold,
        "action_roundtrip_triggered": action_triggered,
        "latent_cycle_p95": latent_p95,
        "latent_cycle_p95_threshold": latent_cycle_p95_threshold,
        "latent_cycle_triggered": latent_triggered,
        "inverse_outlier_fraction": outlier_fraction,
        "inverse_outlier_fraction_threshold": inverse_outlier_fraction_threshold,
        "inverse_outlier_triggered": outlier_triggered,
        "latent_support_radius": latent_support_radius,
    }


def train_async_stage(
    encoder_checkpoint_path: str,
    previous_decoder_checkpoint_path: str,
    replay_snapshot_path: str,
    output_checkpoint_path: str,
    version: int,
    train_steps: int,
    metrics_file: str | None = None,
    inherit_optimizer_state: bool = True,
    decoder_type: str = "flow_matching",
    validation_fraction: float = 0.1,
    validation_batches: int = 4,
    meanflow_min_improvement: float = 1e-4,
    cycle_error_tolerance: float = 1e-6,
) -> dict[str, float | int | bool]:
    """Train one Decoder_n from an immutable replay snapshot.

    This is the existing FM objective (``DecoderFMState.train_step``) without
    success/reward filtering. The latest published encoder is loaded only for
    provenance and validation; decoder optimization does not wait for a new
    encoder version. The current FM
    implementation constructs its own noise latent, so no alternate latent
    target is introduced here.
    """
    if train_steps <= 0:
        raise ValueError("train_steps must be positive.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if validation_batches <= 0:
        raise ValueError("validation_batches must be positive.")
    if meanflow_min_improvement < 0.0 or cycle_error_tolerance < 0.0:
        raise ValueError("Decoder acceptance tolerances must be non-negative.")
    config = TrainingConfig().to_dict() | EnvConfig().to_dict()
    if decoder_type not in ("flow_matching", "meanflow"):
        raise ValueError("decoder_type must be 'flow_matching' or 'meanflow'.")
    with Path(encoder_checkpoint_path).expanduser().open("rb") as file:
        encoder_checkpoint = pickle.load(file)
    required_encoder = {"rlpd_z_actor_params", "rlpd_z_obs_stats", "config"}
    missing = sorted(required_encoder.difference(encoder_checkpoint))
    if missing:
        raise ValueError(f"Fixed encoder checkpoint is missing fields: {missing}")

    with Path(previous_decoder_checkpoint_path).expanduser().open("rb") as file:
        previous = pickle.load(file)
    required_decoder = {"params", "obs_stats", "config", "obs_dim", "action_dim"}
    missing = sorted(required_decoder.difference(previous))
    if missing:
        raise ValueError(f"Previous decoder checkpoint is missing fields: {missing}")

    replay = load_transition_data(replay_snapshot_path)
    states = replay["observations"]
    actions = replay["actions"]
    if len(states) < 2:
        raise ValueError("Decoder stage requires at least two replay transitions.")
    if states.shape[-1] != int(previous["obs_dim"]):
        raise ValueError("Replay observation dimension does not match decoder.")
    if actions.shape[-1] != int(previous["action_dim"]):
        raise ValueError("Replay action dimension does not match decoder.")

    decoder_config = previous["config"]
    state_cls = Decoder1StepFMState if decoder_type == "meanflow" else DecoderFMState
    expected_config = Decoder1StepFMConfig if decoder_type == "meanflow" else DecoderFMConfig
    if not isinstance(decoder_config, expected_config):
        raise ValueError(f"Checkpoint config is not compatible with {decoder_type}.")
    if inherit_optimizer_state and "decoder_opt_state" not in previous:
        raise ValueError("Online decoder continuation requires optimizer state.")
    previous_state = state_cls.init(
        jax.random.PRNGKey(config["seed"] + 1900 + version),
        int(previous["obs_dim"]), int(previous["action_dim"]), decoder_config,
    )
    with jdc.copy_and_mutate(previous_state) as previous_state:
        previous_state.params = previous["params"]
        previous_state.obs_stats = previous["obs_stats"]
        if "decoder_steps" in previous:
            previous_state.steps = previous["decoder_steps"]

    fm_state = state_cls.init(jax.random.PRNGKey(config["seed"] + 2000 + version), int(previous["obs_dim"]), int(previous["action_dim"]), decoder_config)
    with jdc.copy_and_mutate(fm_state) as fm_state:
        fm_state.params = previous["params"]
        fm_state.obs_stats = previous["obs_stats"]
        if inherit_optimizer_state:
            fm_state.opt_state = previous["decoder_opt_state"]
        if "decoder_prng" in previous:
            fm_state.prng = previous["decoder_prng"]
        if "decoder_steps" in previous:
            fm_state.steps = previous["decoder_steps"]
        fm_state.obs_stats = fm_state.obs_stats.update(jnp.asarray(states))

    rng = np.random.default_rng(config["seed"] + version)
    permutation = rng.permutation(len(states))
    validation_size = min(
        len(states) - 1, max(1, int(round(len(states) * validation_fraction)))
    )
    validation_indices = permutation[:validation_size]
    training_indices = permutation[validation_size:]
    validation_states = states[validation_indices]
    validation_actions = actions[validation_indices]
    batch_size = int(decoder_config.batch_size)
    started = time.time()
    metrics: dict[str, Any] = {}
    for step in tqdm(range(train_steps), desc=f"Decoder {version}"):
        indices = training_indices[
            rng.integers(0, len(training_indices), size=batch_size)
        ]
        if decoder_type == "meanflow":
            fm_state, metrics = fm_state.train_step(
                fm_state.steps, jnp.asarray(states[indices]), jnp.asarray(actions[indices])
            )
        else:
            fm_state, metrics = fm_state.train_step(
                jnp.asarray(states[indices]), jnp.asarray(actions[indices])
            )
        if metrics_file and ((step + 1) % 100 == 0 or step + 1 == train_steps):
            append_metrics(metrics_file, {
                "pipeline/version": version,
                "pipeline/decoder_step": step + 1,
                **{f"decoder/{key}": float(np.asarray(value)) for key, value in metrics.items()},
            })

    baseline_metrics = _evaluate_decoder_candidate(
        previous_state, validation_states, validation_actions, decoder_type,
        validation_batches, config["seed"] + version * 100,
    )
    candidate_metrics = _evaluate_decoder_candidate(
        fm_state, validation_states, validation_actions, decoder_type,
        validation_batches, config["seed"] + version * 100,
    )
    meanflow_improvement = (
        baseline_metrics["meanflow_loss"] - candidate_metrics["meanflow_loss"]
    )
    loss_accepted = meanflow_improvement > meanflow_min_improvement
    cycle_accepted = candidate_metrics["cycle_error"] <= (
        baseline_metrics["cycle_error"] + cycle_error_tolerance
    )
    accepted = bool(loss_accepted and cycle_accepted)

    checkpoint = {
        "params": fm_state.params,
        "obs_stats": fm_state.obs_stats,
        "decoder_opt_state": fm_state.opt_state,
        "decoder_prng": fm_state.prng,
        "decoder_steps": fm_state.steps,
        "config": decoder_config,
        "obs_dim": int(previous["obs_dim"]),
        "action_dim": int(previous["action_dim"]),
        "env_name": config["env_name"],
        "decoder_type": decoder_type,
        "version": version,
        "fixed_encoder_checkpoint": str(Path(encoder_checkpoint_path).resolve()),
        "previous_decoder_checkpoint": str(Path(previous_decoder_checkpoint_path).resolve()),
        "train_steps": train_steps,
        "inherited_optimizer_state": inherit_optimizer_state,
        "final_loss": float(np.asarray(metrics.get("loss", np.nan))),
        "accepted": accepted,
        "validation_meanflow_loss": candidate_metrics["meanflow_loss"],
        "baseline_validation_meanflow_loss": baseline_metrics["meanflow_loss"],
        "meanflow_improvement": meanflow_improvement,
        "meanflow_min_improvement": meanflow_min_improvement,
        "validation_cycle_error": candidate_metrics["cycle_error"],
        "baseline_validation_cycle_error": baseline_metrics["cycle_error"],
        "cycle_error_tolerance": cycle_error_tolerance,
        "validation_size": validation_size,
        "wall_time_seconds": time.time() - started,
    }
    atomic_pickle_dump(checkpoint, output_checkpoint_path)
    acceptance_metrics: dict[str, float | int | bool] = {
        "accepted": accepted,
        "train_steps": train_steps,
        "validation_size": validation_size,
        "baseline_meanflow_loss": baseline_metrics["meanflow_loss"],
        "candidate_meanflow_loss": candidate_metrics["meanflow_loss"],
        "meanflow_improvement": meanflow_improvement,
        "baseline_cycle_error": baseline_metrics["cycle_error"],
        "candidate_cycle_error": candidate_metrics["cycle_error"],
    }
    if metrics_file:
        append_metrics(metrics_file, {
            "pipeline/version": version,
            **{
                f"decoder/acceptance_{key}": value
                for key, value in acceptance_metrics.items()
            },
        })
    return acceptance_metrics


def _evaluate_decoder_candidate(
    state: DecoderFMState | Decoder1StepFMState,
    states: np.ndarray,
    actions: np.ndarray,
    decoder_type: str,
    evaluation_batches: int,
    seed: int,
) -> dict[str, float]:
    """Compare candidates with identical held-out samples and random draws."""
    rng = np.random.default_rng(seed)
    batch_size = min(int(state.config.batch_size), len(states))
    losses: list[float] = []
    cycle_errors: list[float] = []
    for batch_index in range(evaluation_batches):
        indices = rng.integers(0, len(states), size=batch_size)
        batch_obs = jnp.asarray(states[indices])
        batch_actions = jnp.asarray(actions[indices])
        key = jax.random.PRNGKey(seed + batch_index)
        if decoder_type == "meanflow":
            eps_key, time_key, decode_key = jax.random.split(key, 3)
            eps = jax.random.normal(eps_key, batch_actions.shape)
            t, r = state.sample_t_r(time_key, batch_size)
            _, meanflow_loss, _ = state.compute_meanflow_loss(
                state.steps,
                state._normalize_obs(batch_obs),
                batch_actions,
                eps,
                t,
                r,
            )
            loss = meanflow_loss
        else:
            eps_key, time_key, decode_key = jax.random.split(key, 3)
            eps = jax.random.normal(
                eps_key,
                (batch_size, state.config.n_samples_per_action, actions.shape[-1]),
            )
            t = jax.random.uniform(
                time_key,
                (batch_size, state.config.n_samples_per_action, 1),
            )
            obs_norm = (
                (batch_obs - state.obs_stats.mean) / (state.obs_stats.std + 1e-8)
                if state.config.normalize_observations else batch_obs
            )
            loss = jnp.mean(
                state.compute_cfm_loss(obs_norm, batch_actions, eps, t)
            )
        latents = state.inverse_fm_batch(batch_obs, batch_actions)
        reconstructed = state.sample_action_from_z(
            batch_obs, latents, decode_key, deterministic=True
        )
        # Mean over both batch and action dimensions makes this metric
        # independent of the action-space dimensionality.
        cycle_error = jnp.mean(jnp.square(reconstructed - batch_actions))
        losses.append(float(np.asarray(loss)))
        cycle_errors.append(float(np.asarray(cycle_error)))
    return {
        "meanflow_loss": float(np.mean(losses)),
        "cycle_error": float(np.mean(cycle_errors)),
    }

def train_fm(
    data_path: str = "data/ppo_training_data_WalkerWalk_20250928_212057.pkl",
    output_dir: str = "fm_models",
    stage: int = 0,
    global_epoch_offset: int = 0,
    wandb_run_id: str | None = None,
    wandb_run_name: str | None = None,
    metrics_file: str | None = None,
    stage_init_before_training: bool = True,
) -> None:
    """Train Flow Matching model on collected PPO data.

    Args:
        data_path: Path to PPO data pickle file
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        validation_split: Fraction of data for validation
        max_samples: Maximum number of samples to use (None for all)
        episode_length: Length of each episode for grouping
        reward_percentile: Keep episodes with reward >= this percentile (0-1)
        min_episode_reward: If set, keep episodes with total reward >= this value
        hybrid_sampling: Enable hybrid sampling (mix high-quality + random coverage)
        high_quality_ratio: In hybrid mode, ratio of high-quality samples (0-1)
        high_quality_percentile: In hybrid mode, percentile for high-quality (0-1)
        output_dir: Directory to save models
        seed: Random seed
        hidden_size: Size of hidden layers (default: 64)
        num_layers: Number of hidden layers (default: 4)
    """
    config = TrainingConfig().to_dict() | EnvConfig().to_dict()
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # When metrics_file is provided by the pipeline, only the parent process
    # owns W&B and this process emits JSONL events. Standalone execution keeps
    # direct W&B logging for backward compatibility.
    wandb_run = None
    if config["wandb_enabled"] and metrics_file is None:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "wandb is required when wandb_enabled=True."
            ) from error
        wandb_run = wandb.init(
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            name=wandb_run_name or f"decoder_fm_stage_{stage}",
            id=wandb_run_id,
            resume="allow" if wandb_run_id else None,
            group=config["wandb_group"] or wandb_run_id,
            tags=list(config["wandb_tags"]),
            mode=config["wandb_mode"],
            config={**config, "pipeline_run_id": wandb_run_id},
        )
        wandb_run.define_metric("pipeline/decoder_step")
        wandb_run.define_metric("decoder/*", step_metric="pipeline/decoder_step")

    # Load data
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    resume_checkpoint = None
    resume_checkpoint_path = None
    if not stage_init_before_training:
        checkpoint_from_data = data.get("fm_model")
        if stage > 0 and not checkpoint_from_data:
            raise ValueError(
                "stage_init_before_training=False requires collected data with an "
                f"'fm_model' checkpoint path for stage {stage}."
            )

        if checkpoint_from_data:
            candidate_path = Path(checkpoint_from_data).expanduser()
            if not candidate_path.is_file():
                raise FileNotFoundError(
                    f"Decoder checkpoint referenced by collected data was not found: "
                    f"{candidate_path}"
                )
            with candidate_path.open("rb") as f:
                candidate_checkpoint = pickle.load(f)
            if not isinstance(candidate_checkpoint, dict):
                raise ValueError(
                    f"Decoder checkpoint must contain a dictionary: {candidate_path}"
                )

            # In a pure-online run, stage 0 data is collected with the identity
            # decoder. That checkpoint is only a bootstrap policy, not a trained
            # state to resume, so initialize the trainable FM network normally.
            # Offline stage-0 checkpoints and every later-stage checkpoint are
            # still restored directly.
            should_resume_decoder = stage > 0 or not candidate_checkpoint.get(
                "is_identity", False
            )
            if should_resume_decoder:
                required_fields = {
                    "params", "obs_stats", "config", "obs_dim", "action_dim"
                }
                missing_fields = sorted(
                    required_fields.difference(candidate_checkpoint)
                )
                if missing_fields:
                    raise ValueError(
                        f"Decoder resume checkpoint {candidate_path} is missing "
                        f"required fields: {missing_fields}"
                    )
                resume_checkpoint_path = candidate_path
                resume_checkpoint = candidate_checkpoint
                print(f"Continuing decoder training from: {resume_checkpoint_path}")
            else:
                print(
                    "Initializing decoder normally for online stage 0; identity "
                    "checkpoint is used only for data collection."
                )

    states = data["states"]
    actions = data["actions"]

    # Episode-based filtering using rewards
    if "rewards" in data and (config['fm_reward_percentile'] > 0 or config['fm_min_episode_reward'] is not None or config['fm_hybrid_sampling']):
        rewards = data["rewards"]

        # Group data by episodes
        n_episodes = len(states) // config['episode_length']
        if len(states) % config['episode_length'] != 0:
            trim_to = n_episodes * config['episode_length']
            states = states[:trim_to]
            actions = actions[:trim_to]
            rewards = rewards[:trim_to]

        episodes = []
        for i in range(n_episodes):
            start = i * config['episode_length']
            end = start + config['episode_length']
            episode_total_reward = rewards[start:end].sum()
            episodes.append({
                'idx': i,
                'start': start,
                'end': end,
                'total_reward': float(episode_total_reward)
            })

        episode_rewards = np.array([ep['total_reward'] for ep in episodes])

        # Apply filtering or hybrid sampling
        if config['fm_hybrid_sampling']:
            hq_threshold = np.percentile(episode_rewards, [config['fm_high_quality_percentile']] * 100)
            high_quality_episodes = [ep for ep in episodes if ep['total_reward'] >= hq_threshold]

            n_high_quality = int(len(episodes) * config['fm_high_quality_ratio'])
            n_coverage = len(episodes) - n_high_quality

            if len(high_quality_episodes) >= n_high_quality:
                sampled_hq = np.random.choice(len(high_quality_episodes), n_high_quality, replace=False)
                selected_hq_episodes = [high_quality_episodes[i] for i in sampled_hq]
            else:
                sampled_hq = np.random.choice(len(high_quality_episodes), n_high_quality, replace=True)
                selected_hq_episodes = [high_quality_episodes[i] for i in sampled_hq]

            sampled_coverage = np.random.choice(len(episodes), n_coverage, replace=False)
            selected_coverage_episodes = [episodes[i] for i in sampled_coverage]

            keep_episodes = selected_hq_episodes + selected_coverage_episodes

        elif config['fm_min_episode_reward'] is not None:
            keep_episodes = [ep for ep in episodes if ep['total_reward'] >= config['fm_min_episode_reward']]
        else:
            threshold = np.percentile(episode_rewards, config['fm_reward_percentile'] * 100)
            keep_episodes = [ep for ep in episodes if ep['total_reward'] >= threshold]

        # Rebuild data from kept episodes
        keep_indices = []
        for ep in keep_episodes:
            keep_indices.extend(range(ep['start'], ep['end']))

        states = states[keep_indices]
        actions = actions[keep_indices]
        rewards = rewards[keep_indices]
    else:
        if "rewards" in data:
            rewards = data["rewards"]

    # Optionally subsample data for faster training
    if config['fm_max_samples'] is not None and len(states) > config['fm_max_samples']:
        sample_indices = np.random.choice(len(states), config['fm_max_samples'], replace=False)
        states = states[sample_indices]
        actions = actions[sample_indices]
        if "rewards" in data:
            rewards = rewards[sample_indices]

    # Split data
    n_samples = len(states)
    n_train = int(n_samples * (1 - config['fm_validation_split']))
    indices = np.random.permutation(n_samples)

    train_states = states[indices[:n_train]]
    train_actions = actions[indices[:n_train]]
    val_states = states[indices[n_train:]]
    val_actions = actions[indices[n_train:]]

    # Initialize FM model
    obs_dim = states.shape[1]
    action_dim = actions.shape[1]

    # Build hidden dims from parameters
    hidden_dims = tuple([config['fm_hidden_size']] * config['fm_num_layers'])

    if resume_checkpoint is None:
        decoder_config = DecoderFMConfig(
            flow_steps=10,
            timestep_embed_dim=8,  # FPO uses 8
            hidden_dims=hidden_dims,  # Configurable network size
            policy_output_scale=1.0,  # Changed to 1.0 for supervised learning
            learning_rate=config['fm_learning_rate'],
            batch_size=config['fm_batch_size'],
            num_epochs=config['fm_num_epochs'],
            n_samples_per_action=config['fm_n_samples_per_action'],  # FPO's actual default
            normalize_observations=True,
            sde_sigma=0.0,
            feather_std=0.0,
        )
    else:
        if int(resume_checkpoint["obs_dim"]) != obs_dim:
            raise ValueError(
                f"Resume checkpoint obs_dim={resume_checkpoint['obs_dim']} does not "
                f"match collected data obs_dim={obs_dim}."
            )
        if int(resume_checkpoint["action_dim"]) != action_dim:
            raise ValueError(
                f"Resume checkpoint action_dim={resume_checkpoint['action_dim']} does "
                f"not match collected data action_dim={action_dim}."
            )
        decoder_config = resume_checkpoint["config"]
        if not isinstance(decoder_config, DecoderFMConfig):
            raise TypeError(
                f"Resume checkpoint {resume_checkpoint_path} has an invalid "
                f"decoder config type: {type(decoder_config).__name__}."
            )

    prng = jax.random.PRNGKey(config['seed'])
    fm_state = DecoderFMState.init(prng, obs_dim, action_dim, decoder_config)

    # Online optimizers are independent from offline training. Offline
    # checkpoints restore only model parameters and observation statistics;
    # DecoderFMState.init provides fresh online optimizer/PRNG/step state.
    resume_is_offline = bool(
        resume_checkpoint is not None
        and (
            resume_checkpoint.get("is_frozen_offline", False)
            or resume_checkpoint.get("offline_checkpoint_type") is not None
            or str(resume_checkpoint.get("checkpoint_format", "")).startswith(
                "gorl_offline"
            )
        )
    )
    with jdc.copy_and_mutate(fm_state) as fm_state:
        if resume_checkpoint is not None:
            fm_state.params = resume_checkpoint["params"]
            fm_state.obs_stats = resume_checkpoint["obs_stats"]
            # Offline optimizer state is never inherited. A previous online
            # stage may retain its own optimizer when continuation is requested.
            if not resume_is_offline and "decoder_opt_state" in resume_checkpoint:
                fm_state.opt_state = resume_checkpoint["decoder_opt_state"]
            if not resume_is_offline and "decoder_prng" in resume_checkpoint:
                fm_state.prng = resume_checkpoint["decoder_prng"]
            if not resume_is_offline and "decoder_steps" in resume_checkpoint:
                fm_state.steps = resume_checkpoint["decoder_steps"]
        fm_state.obs_stats = fm_state.obs_stats.update(jnp.array(train_states))

    # Preserve the encoder train state in decoder checkpoints. The pipeline only
    # passes the latest decoder checkpoint into the next encoder stage, so these
    # fields allow stage_init_before_training=False to resume PPO or RLPD.
    carried_encoder_fields = {}
    encoder_checkpoint_path = data.get("encoder_checkpoint") or data.get(
        "ppo_z_checkpoint"
    )
    if encoder_checkpoint_path:
        encoder_checkpoint_path = Path(encoder_checkpoint_path).expanduser()
        if not encoder_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Encoder checkpoint referenced by collected data was not found: "
                f"{encoder_checkpoint_path}"
            )
        with encoder_checkpoint_path.open("rb") as f:
            encoder_checkpoint = pickle.load(f)
        encoder_algorithm = data.get("encoder_algorithm")
        if encoder_algorithm is None:
            encoder_algorithm = (
                "rlpd" if "rlpd_z_actor_params" in encoder_checkpoint else "ppo"
            )
        if encoder_algorithm == "rlpd":
            required_encoder_fields = {
                "rlpd_z_actor_params",
                "rlpd_z_critic_params",
                "rlpd_z_target_critic_params",
                "rlpd_z_log_temperature",
                "rlpd_z_obs_stats",
            }
            encoder_field_names = (
                "rlpd_z_actor_params",
                "rlpd_z_critic_params",
                "rlpd_z_target_critic_params",
                "rlpd_z_log_temperature",
                "rlpd_z_actor_opt_state",
                "rlpd_z_critic_opt_state",
                "rlpd_z_temperature_opt_state",
                "rlpd_z_obs_stats",
                "rlpd_z_prng",
                "rlpd_z_steps",
            )
        elif encoder_algorithm == "ppo":
            required_encoder_fields = {"ppo_z_params", "ppo_z_obs_stats"}
            encoder_field_names = (
                "ppo_z_params",
                "ppo_z_obs_stats",
                "ppo_z_anchor_policy",
                "ppo_z_anchor_obs_stats",
                "ppo_z_opt_state",
                "ppo_z_prng",
                "ppo_z_steps",
            )
        else:
            raise ValueError(
                f"Unsupported encoder_algorithm in collected data: {encoder_algorithm!r}"
            )
        missing_encoder_fields = sorted(
            required_encoder_fields.difference(encoder_checkpoint)
        )
        if missing_encoder_fields:
            raise ValueError(
                f"Encoder checkpoint {encoder_checkpoint_path} is missing required "
                f"fields: {missing_encoder_fields}"
            )
        for key in encoder_field_names:
            if key in encoder_checkpoint:
                carried_encoder_fields[key] = encoder_checkpoint[key]

    # Training loop
    n_batches = n_train // config['fm_batch_size']
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0
    patience = 20  # Early stopping patience

    for epoch in range(config['fm_num_epochs']):
        # Training
        epoch_losses = []
        epoch_metrics = []

        # Shuffle training data
        perm = np.random.permutation(n_train)

        for batch_idx in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{config['fm_num_epochs']}"):
            # Get batch
            start_idx = batch_idx * config['fm_batch_size']
            end_idx = start_idx + config['fm_batch_size']
            batch_indices = perm[start_idx:end_idx]

            batch_obs = jnp.array(train_states[batch_indices])
            batch_actions = jnp.array(train_actions[batch_indices])

            # Training step
            fm_state, metrics = fm_state.train_step(batch_obs, batch_actions)

            epoch_losses.append(float(metrics["loss"]))
            epoch_metrics.append(metrics)

        # Compute epoch statistics
        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)

        # Validation
        val_batch_losses = []
        n_val_batches = min(50, len(val_states) // config['fm_batch_size'])  # Increased validation coverage

        for i in range(n_val_batches):
            start_idx = i * config['fm_batch_size']
            end_idx = start_idx + config['fm_batch_size']
            batch_obs = jnp.array(val_states[start_idx:end_idx])
            batch_actions = jnp.array(val_actions[start_idx:end_idx])

            # Use CFM loss for validation (same as training)
            # Normalize observations
            if fm_state.config.normalize_observations:
                obs_norm = (batch_obs - fm_state.obs_stats.mean) / (fm_state.obs_stats.std + 1e-8)
            else:
                obs_norm = batch_obs

            # Sample noise and timesteps
            prng_val_eps, prng_val_t, prng = jax.random.split(fm_state.prng, 3)
            val_eps = jax.random.normal(
                prng_val_eps,
                (config['fm_batch_size'], fm_state.config.n_samples_per_action, action_dim)
            )
            val_t = jax.random.uniform(
                prng_val_t,
                (config['fm_batch_size'], fm_state.config.n_samples_per_action, 1)
            )

            # Compute CFM loss
            cfm_loss = fm_state.compute_cfm_loss(
                obs_norm,
                batch_actions,
                val_eps,
                val_t
            )
            val_loss = jnp.mean(cfm_loss)
            val_batch_losses.append(float(val_loss))

        val_loss = np.mean(val_batch_losses)
        val_losses.append(val_loss)

        # Save best model
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            # Save checkpoint
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = output_path / f"fm_model_best_{timestamp}.pkl"

            checkpoint = {
                "params": fm_state.params,  # Only save parameters
                "obs_stats": fm_state.obs_stats,
                "decoder_opt_state": fm_state.opt_state,
                "decoder_prng": fm_state.prng,
                "decoder_steps": fm_state.steps,
                "config": decoder_config,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "resumed_from": (
                    str(resume_checkpoint_path)
                    if resume_checkpoint_path is not None
                    else None
                ),
                **carried_encoder_fields,
            }

            with open(checkpoint_file, "wb") as f:
                pickle.dump(checkpoint, f)

            patience_counter = 0
        else:
            patience_counter += 1

        epoch_log = {
                "pipeline/decoder_step": global_epoch_offset + epoch + 1,
                "pipeline/stage": stage,
                "decoder/epoch": epoch + 1,
                "decoder/train_loss": float(train_loss),
                "decoder/val_loss": float(val_loss),
                "decoder/best_val_loss": float(best_val_loss),
                "decoder/patience_counter": patience_counter,
                "decoder/improved": int(improved),
        }
        if wandb_run is not None:
            wandb_run.log(epoch_log)
        elif metrics_file is not None:
            append_metrics(metrics_file, epoch_log)

        if patience_counter >= patience:
            break

    # Save final model
    final_file = output_path / f"fm_model_final_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    final_checkpoint = {
        "params": fm_state.params,  # Only save parameters
        "obs_stats": fm_state.obs_stats,
        "decoder_opt_state": fm_state.opt_state,
        "decoder_prng": fm_state.prng,
        "decoder_steps": fm_state.steps,
        "config": decoder_config,
        "epoch": config['fm_num_epochs'],
        "train_loss": train_losses[-1],
        "val_loss": val_losses[-1],
        "train_history": train_losses,
        "val_history": val_losses,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "resumed_from": (
            str(resume_checkpoint_path)
            if resume_checkpoint_path is not None
            else None
        ),
        **carried_encoder_fields,
    }

    with open(final_file, "wb") as f:
        pickle.dump(final_checkpoint, f)

    completed_epoch = len(train_losses)
    completion_log = {
            "pipeline/decoder_step": global_epoch_offset + completed_epoch,
            "pipeline/stage": stage,
            "decoder/completed": 1,
            "decoder/completed_epochs": completed_epoch,
            "decoder/final_train_loss": float(train_losses[-1]),
            "decoder/final_val_loss": float(val_losses[-1]),
            "decoder/final_best_val_loss": float(best_val_loss),
            "stage/completed": 1,
            f"stage_{stage}/decoder_completed": 1,
            f"stage_{stage}/completed": 1,
    }
    if wandb_run is not None:
        wandb_run.log(completion_log)
        wandb_run.finish()
    elif metrics_file is not None:
        append_metrics(metrics_file, completion_log)

    print(f"Decoder (FM) done: loss={best_val_loss:.4f}, output={checkpoint_file}")


if __name__ == "__main__":
    tyro.cli(train_fm)
