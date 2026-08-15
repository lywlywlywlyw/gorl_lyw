"""Frozen-decoder offline training compatible with the online GoRL FM stack.

The method is the same two-stage procedure as ``run_offline_fm_frozen.py``:

1. Train a conditional flow-matching decoder to convergence and freeze it.
2. Invert dataset actions through the frozen decoder and train the latent
   encoder with IQL advantage-weighted behavior cloning.

This file is deliberately self-contained with respect to the old offline
scripts.  It imports only the same production network/state implementations
used by ``scripts/run_gorl_fm.py`` and its components.

The final pickle has both:

* top-level ``params/obs_stats/config/obs_dim/action_dim`` fields accepted by
  ``scripts/components/train_encoder_ppo.py`` as an FM decoder checkpoint;
* ``ppo_z_params/ppo_z_obs_stats/config`` fields matching online encoder
  checkpoints and accepted by ``scripts/components/collect_data_fm.py``.

The decoder checkpoints are marked with the stable ``gorl_fm_decoder`` format
so they can be passed directly to ``scripts/run_gorl_fm.py`` via
``--use-offline-checkpoint --offline-checkpoint-path ...``.

Example:
    python run_offline_fm_frozen_robomimic.py

All environment, frozen-FM, IQL, checkpoint-metadata, and logging parameters
are sourced from ``envs.robomimic.offline_config``.
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow this root-level script to import the local ``src/flow_policy`` package
# when invoked directly with ``python run_offline_fm_frozen_robomimic.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import jax
import jax_dataclasses as jdc
import numpy as np
import optax
import tyro
from jax import Array
from jax import numpy as jnp
from tqdm import trange

from envs.robomimic.RobomimicEnv import RobomimicEnv
from envs.robomimic.offline_config.env_config import EnvConfig
from envs.robomimic.offline_config.training_config import TrainingConfig
from flow_policy import encoder_ppo, math_utils, networks
from flow_policy.decoder_fm import DecoderFMConfig, DecoderFMState


PyTree = Any


class ConfigView(dict[str, Any]):
    """Dictionary config with attribute access for the training implementation."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def build_config(training_config: TrainingConfig | None = None) -> ConfigView:
    """Compose the robomimic frozen-IQL configuration from offline configs."""
    training_config = training_config or TrainingConfig()
    config = ConfigView(training_config.to_dict() | EnvConfig().to_dict())
    # Internal aliases keep the implementation names aligned with the generic
    # reference script while all values remain owned by offline_config.
    config.update(
        max_samples=config["fm_max_samples"],
        decoder_learning_rate=config["fm_learning_rate"],
        decoder_hidden_size=config["fm_hidden_size"],
        decoder_num_layers=config["fm_num_layers"],
        decoder_batch_size=config["fm_batch_size"],
        decoder_max_epochs=config["fm_num_epochs"],
        decoder_checkpoint_interval=config["fm_checkpoint_interval"],
        decoder_validation_fraction=config["fm_validation_split"],
        flow_steps=config["fm_flow_steps"],
        timestep_embed_dim=config["fm_timestep_embed_dim"],
        n_fm_samples_per_action=config["fm_n_samples_per_action"],
        decoder_min_epochs=config["fm_min_epochs"],
        decoder_patience=config["fm_patience"],
        decoder_min_delta=config["fm_min_delta"],
        decoder_eval_batches=config["fm_eval_batches"],
        latent_inverse_steps=config["fm_latent_inverse_steps"],
    )
    return config


@dataclass
class ReplayBuffer:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    masks: np.ndarray

    def __len__(self) -> int:
        return len(self.observations)


class WandbLogger:
    def __init__(self, config: ConfigView):
        self.run = None
        if not config.wandb_enabled or config.wandb_mode == "disabled":
            return
        try:
            import wandb
        except ImportError as error:
            raise ImportError("W&B logging requires the wandb package.") from error
        self.run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            group=config.wandb_group,
            name=config.wandb_name,
            mode=config.wandb_mode,
            config={**dict(config), "method": "frozen_decoder"},
            tags=["frozen_decoder", "online_compatible"],
        )

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def flatten_robomimic_observations(
    obs_group: Any, obs_keys: list[str]
) -> np.ndarray:
    """Flatten low-dimensional robomimic observations in environment key order."""
    arrays = [np.asarray(obs_group[key], dtype=np.float32) for key in obs_keys]
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValueError("Robomimic observation arrays have inconsistent lengths.")
    return np.concatenate(
        [array.reshape(len(array), -1) for array in arrays], axis=-1
    )


def load_replay_buffer(
    config: ConfigView, environment: RobomimicEnv
) -> ReplayBuffer:
    """Load transitions from the robomimic HDF5 dataset used by the env."""
    try:
        import h5py
    except ImportError as error:
        raise ImportError("Loading robomimic datasets requires h5py.") from error

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    dataset_path = Path(config.dataset_path).expanduser()
    with h5py.File(dataset_path, "r") as dataset:
        if "data" not in dataset:
            raise KeyError(f"{dataset_path} does not contain a robomimic data group.")
        for demo_key in sorted(dataset["data"].keys()):
            demo = dataset[f"data/{demo_key}"]
            if "obs" not in demo or "actions" not in demo:
                raise KeyError(f"Demo {demo_key} is missing obs or actions.")
            obs = flatten_robomimic_observations(demo["obs"], environment.obs_keys)
            action = np.asarray(demo["actions"], dtype=np.float32)
            reward = np.asarray(
                demo["rewards"] if "rewards" in demo else np.zeros(len(action)),
                dtype=np.float32,
            ).reshape(-1)
            done = np.asarray(
                demo["dones"] if "dones" in demo else np.zeros(len(action)),
                dtype=np.float32,
            ).reshape(-1)
            if "next_obs" in demo:
                next_obs = flatten_robomimic_observations(
                    demo["next_obs"], environment.obs_keys
                )
            else:
                if len(obs) < 2:
                    continue
                next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)
                done[-1] = 1.0
            if not (len(obs) == len(action) == len(reward) == len(next_obs) == len(done)):
                raise ValueError(f"Demo {demo_key} has inconsistent transition lengths.")
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            masks.append(1.0 - done)

    if not observations:
        raise ValueError(f"No usable robomimic demos found in {dataset_path}.")
    buffer = ReplayBuffer(
        np.concatenate(observations),
        np.concatenate(actions),
        np.concatenate(rewards),
        np.concatenate(next_observations),
        np.concatenate(masks),
    )
    if config.max_samples is not None and len(buffer) > config.max_samples:
        rng = np.random.default_rng(config.seed)
        indices = rng.choice(len(buffer), config.max_samples, replace=False)
        buffer = ReplayBuffer(
            buffer.observations[indices],
            buffer.actions[indices],
            buffer.rewards[indices],
            buffer.next_observations[indices],
            buffer.masks[indices],
        )
    return buffer


def validate_config(config: ConfigView) -> None:
    if config.decoder_max_epochs < 1:
        raise ValueError("decoder_max_epochs must be positive.")
    if not 1 <= config.decoder_min_epochs <= config.decoder_max_epochs:
        raise ValueError("decoder_min_epochs must be within decoder epochs.")
    if config.decoder_patience < 1:
        raise ValueError("decoder_patience must be positive.")
    if not 0.0 < config.decoder_validation_fraction < 1.0:
        raise ValueError("decoder_validation_fraction must be in (0, 1).")
    if min(
        config.batch_size,
        config.decoder_batch_size,
        config.encoder_iql_steps,
        config.latent_inverse_steps,
        config.comparison_samples,
        config.checkpoint_interval,
        config.decoder_checkpoint_interval,
    ) < 1:
        raise ValueError("Batch sizes and step/sample counts must be positive.")
    if config.wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline, or disabled.")


def make_dataset_environment(
    config: ConfigView,
) -> tuple[RobomimicEnv, str]:
    """Create robomimic exactly as in ``scripts/run_gorl_fm.py``."""
    environment = RobomimicEnv(dataset_path=config.dataset_path, reward_shaping=config.dense_reward)
    return environment, config.env_name


def make_encoder_config(
    config: ConfigView,
    action_dim: int,
    episode_length: int,
) -> encoder_ppo.EncoderConfig:
    """Build the same online checkpoint metadata as the reference script."""
    return encoder_ppo.EncoderConfig(
        action_repeat=1,
        batch_size=config.batch_size,
        discounting=config.discount,
        entropy_cost=0.0,
        episode_length=episode_length,
        learning_rate=config.actor_learning_rate,
        normalize_observations=True,
        num_envs=1,
        num_evals=1,
        num_minibatches=1,
        num_timesteps=config.online_num_timesteps,
        num_updates_per_batch=1,
        reward_scaling=1.0,
        unroll_length=1,
        z_dim=action_dim,
        clipping_epsilon=config.online_clipping_epsilon,
        z_regularization=config.online_z_regularization,
        max_grad_norm=config.online_max_grad_norm,
        use_tanh_jacobian_for_z=config.online_use_tanh_jacobian_for_z,
    )


def split_indices(
    size: int, validation_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if size < 2:
        raise ValueError("At least two transitions are required.")
    permutation = rng.permutation(size)
    validation_size = min(size - 1, max(1, int(size * validation_fraction)))
    return permutation[validation_size:], permutation[:validation_size]


def inverse_fm_batch(
    decoder: DecoderFMState,
    observations: Array,
    actions: Array,
    num_steps: int,
) -> Array:
    obs_norm = (observations - decoder.obs_stats.mean) / (
        decoder.obs_stats.std + 1e-8
    )
    times = jnp.linspace(0.0, 1.0, num_steps + 1)

    def step(x_t: Array, pair: tuple[Array, Array]) -> tuple[Array, None]:
        current, following = pair
        t = jnp.full((*x_t.shape[:-1], 1), current)
        velocity = decoder.flow_forward(obs_norm, x_t, decoder.embed_timestep(t))
        return x_t + (following - current) * velocity, None

    latent, _ = jax.lax.scan(
        step, actions, (times[:-1], times[1:])
    )
    return latent


def forward_fm_batch(
    decoder: DecoderFMState, observations: Array, latents: Array
) -> Array:
    obs_norm = (observations - decoder.obs_stats.mean) / (
        decoder.obs_stats.std + 1e-8
    )
    schedule = decoder.get_schedule()

    def step(x_t: Array, pair: tuple[Array, Array]) -> tuple[Array, None]:
        current, following = pair
        t = jnp.full((*x_t.shape[:-1], 1), current)
        velocity = decoder.flow_forward(obs_norm, x_t, decoder.embed_timestep(t))
        return x_t + (following - current) * velocity, None

    actions, _ = jax.lax.scan(
        step, latents, (schedule.t_current, schedule.t_next)
    )
    return actions


def build_latent_targets(
    decoder: DecoderFMState,
    buffer: ReplayBuffer,
    batch_size: int,
    inverse_steps: int,
) -> np.ndarray:
    invert = jax.jit(
        lambda obs, act: inverse_fm_batch(decoder, obs, act, inverse_steps)
    )
    chunks = []
    for start in trange(
        0, len(buffer), batch_size, desc="Invert actions", leave=False
    ):
        end = min(start + batch_size, len(buffer))
        chunks.append(
            np.asarray(
                invert(
                    jnp.asarray(buffer.observations[start:end]),
                    jnp.asarray(buffer.actions[start:end]),
                )
            )
        )
    return np.concatenate(chunks)


def decoder_validation_loss(
    decoder: DecoderFMState,
    buffer: ReplayBuffer,
    indices: np.ndarray,
    max_batches: int,
    key: Array,
) -> tuple[float, Array]:
    losses = []
    for batch_number, start in enumerate(
        range(0, len(indices), decoder.config.batch_size)
    ):
        if batch_number >= max_batches:
            break
        batch = indices[start : start + decoder.config.batch_size]
        obs = jnp.asarray(buffer.observations[batch])
        actions = jnp.asarray(buffer.actions[batch])
        obs_norm = (obs - decoder.obs_stats.mean) / (
            decoder.obs_stats.std + 1e-8
        )
        key, eps_key, time_key = jax.random.split(key, 3)
        eps = jax.random.normal(
            eps_key,
            (len(batch), decoder.config.n_samples_per_action, actions.shape[-1]),
        )
        times = jax.random.uniform(
            time_key, (len(batch), decoder.config.n_samples_per_action, 1)
        )
        losses.append(
            float(jnp.mean(decoder.compute_cfm_loss(obs_norm, actions, eps, times)))
        )
    return float(np.mean(losses)), key


def decoder_metrics(
    decoder: DecoderFMState,
    buffer: ReplayBuffer,
    indices: np.ndarray,
    inverse_steps: int,
) -> tuple[dict[str, float], np.ndarray]:
    obs = jnp.asarray(buffer.observations[indices])
    actions = jnp.asarray(buffer.actions[indices])
    latents = jax.jit(
        lambda o, a: inverse_fm_batch(decoder, o, a, inverse_steps)
    )(obs, actions)
    reconstructed = jax.jit(
        lambda o, z: forward_fm_batch(decoder, o, z)
    )(obs, latents)
    latent_std = jnp.std(latents, axis=0)
    metrics = {
        "decoder/cycle_action_mse": float(
            jnp.mean(jnp.square(reconstructed - actions))
        ),
        "decoder/cycle_action_mae": float(
            jnp.mean(jnp.abs(reconstructed - actions))
        ),
        "latent/mean_abs": float(jnp.mean(jnp.abs(jnp.mean(latents, axis=0)))),
        "latent/std_mean": float(jnp.mean(latent_std)),
        "latent/std_error": float(jnp.mean(jnp.abs(latent_std - 1.0))),
        "latent/norm_mean": float(jnp.mean(jnp.linalg.norm(latents, axis=-1))),
        "latent/max_abs": float(jnp.max(jnp.abs(latents))),
    }
    return metrics, np.asarray(latents)


def expectile_loss(diff: Array, expectile: float) -> Array:
    weight = jnp.where(diff > 0, expectile, 1.0 - expectile)
    return weight * jnp.square(diff)


def polyak_update(params: PyTree, targets: PyTree, tau: float) -> PyTree:
    return jax.tree.map(
        lambda param, target: tau * param + (1.0 - tau) * target,
        params,
        targets,
    )


def make_iql_update(
    config: ConfigView,
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
    value_optimizer: optax.GradientTransformation,
):
    @jax.jit
    def update(
        actor_params,
        actor_opt_state,
        q1_params,
        q2_params,
        critic_opt_state,
        value_params,
        value_opt_state,
        target_q1_params,
        target_q2_params,
        obs,
        actions,
        rewards,
        next_obs,
        masks,
        latent_actions,
    ):
        target_q = jnp.minimum(
            networks.q_mlp_fwd(target_q1_params, obs, actions),
            networks.q_mlp_fwd(target_q2_params, obs, actions),
        )

        def value_loss_fn(params):
            value = networks.value_mlp_fwd(params, obs)
            loss = jnp.mean(expectile_loss(target_q - value, config.expectile))
            return loss, (value, target_q - value)

        (value_loss, (value, advantage)), value_grads = jax.value_and_grad(
            value_loss_fn, has_aux=True
        )(value_params)
        value_updates, value_opt_state = value_optimizer.update(
            value_grads, value_opt_state, value_params
        )
        value_params = optax.apply_updates(value_params, value_updates)

        new_value = networks.value_mlp_fwd(value_params, obs)
        actor_advantage = jax.lax.stop_gradient(target_q - new_value)
        advantage_weight = jnp.minimum(
            jnp.exp(actor_advantage * config.temperature),
            config.max_adv_weight,
        )

        def actor_loss_fn(params):
            distribution = networks.gaussian_policy_fwd(params, obs)
            log_prob = jnp.sum(distribution.log_prob(latent_actions), axis=-1)
            return -jnp.mean(advantage_weight * log_prob)

        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(actor_params)
        actor_updates, actor_opt_state = actor_optimizer.update(
            actor_grads, actor_opt_state, actor_params
        )
        actor_params = optax.apply_updates(actor_params, actor_updates)

        next_value = jax.lax.stop_gradient(
            networks.value_mlp_fwd(value_params, next_obs)
        )
        bellman_target = rewards + config.discount * masks * next_value

        def critic_loss_fn(params):
            q1, q2 = params
            q1_value = networks.q_mlp_fwd(q1, obs, actions)
            q2_value = networks.q_mlp_fwd(q2, obs, actions)
            loss = jnp.mean(
                jnp.square(q1_value - bellman_target)
                + jnp.square(q2_value - bellman_target)
            )
            return loss, (q1_value, q2_value)

        (critic_loss, (q1_value, q2_value)), critic_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )((q1_params, q2_params))
        critic_updates, critic_opt_state = critic_optimizer.update(
            critic_grads, critic_opt_state, (q1_params, q2_params)
        )
        q1_params, q2_params = optax.apply_updates(
            (q1_params, q2_params), critic_updates
        )
        target_q1_params = polyak_update(
            q1_params, target_q1_params, config.target_update_rate
        )
        target_q2_params = polyak_update(
            q2_params, target_q2_params, config.target_update_rate
        )
        return (
            actor_params,
            actor_opt_state,
            q1_params,
            q2_params,
            critic_opt_state,
            value_params,
            value_opt_state,
            target_q1_params,
            target_q2_params,
            {
                "value_loss": value_loss,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "value": jnp.mean(value),
                "q1": jnp.mean(q1_value),
                "q2": jnp.mean(q2_value),
                "advantage": jnp.mean(advantage),
                "adv_weight": jnp.mean(advantage_weight),
            },
        )

    return update


def policy_metrics(
    actor_params: PyTree,
    q1_params: PyTree,
    q2_params: PyTree,
    value_params: PyTree,
    decoder: DecoderFMState,
    buffer: ReplayBuffer,
    normalized_observations: np.ndarray,
    indices: np.ndarray,
    latent_targets: np.ndarray,
) -> dict[str, float]:
    obs_norm = jnp.asarray(normalized_observations[indices])
    obs_raw = jnp.asarray(buffer.observations[indices])
    data_actions = jnp.asarray(buffer.actions[indices])
    targets = jnp.asarray(latent_targets[indices])
    distribution = networks.gaussian_policy_fwd(actor_params, obs_norm)
    policy_z = distribution.loc
    policy_actions = jax.jit(
        lambda o, z: forward_fm_batch(decoder, o, z)
    )(obs_raw, policy_z)
    q_policy = jnp.minimum(
        networks.q_mlp_fwd(q1_params, obs_norm, policy_actions),
        networks.q_mlp_fwd(q2_params, obs_norm, policy_actions),
    )
    q_data = jnp.minimum(
        networks.q_mlp_fwd(q1_params, obs_norm, data_actions),
        networks.q_mlp_fwd(q2_params, obs_norm, data_actions),
    )
    value = networks.value_mlp_fwd(value_params, obs_norm)
    return {
        "comparison/policy_q": float(jnp.mean(q_policy)),
        "comparison/data_q": float(jnp.mean(q_data)),
        "comparison/policy_q_minus_data_q": float(jnp.mean(q_policy - q_data)),
        "comparison/policy_q_minus_v": float(jnp.mean(q_policy - value)),
        "comparison/policy_action_data_mse": float(
            jnp.mean(jnp.square(policy_actions - data_actions))
        ),
        "encoder/latent_nll": float(
            -jnp.mean(jnp.sum(distribution.log_prob(targets), axis=-1))
        ),
        "encoder/mean_target_mse": float(
            jnp.mean(jnp.square(policy_z - targets))
        ),
        "encoder/scale_mean": float(jnp.mean(distribution.scale)),
    }


def append_metrics(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a") as file:
        file.write(json.dumps(record) + "\n")


def load_checkpoint(path: str | None) -> tuple[dict[str, Any] | None, Path | None]:
    if path is None:
        return None, None
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    with open(checkpoint_path, "rb") as file:
        checkpoint = pickle.load(file)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must contain a dictionary: {checkpoint_path}")
    return checkpoint, checkpoint_path


def checkpoint_iql_step(checkpoint: dict[str, Any], path: Path | None) -> int:
    if "encoder_iql_step" in checkpoint:
        return int(checkpoint["encoder_iql_step"])
    if path is not None:
        match = re.search(r"checkpoint_step_(\d+)", path.name)
        if match is not None:
            return int(match.group(1))
    return int(checkpoint.get("iteration", 0))


def checkpoint_decoder_epoch(
    checkpoint: dict[str, Any], path: Path | None
) -> int:
    if "decoder_epoch" in checkpoint:
        return int(checkpoint["decoder_epoch"])
    if path is not None:
        match = re.search(r"decoder_checkpoint_epoch_(\d+)", path.name)
        if match is not None:
            return int(match.group(1))
    return int(checkpoint.get("epoch", 0))


def is_combined_checkpoint(checkpoint: dict[str, Any]) -> bool:
    checkpoint_type = checkpoint.get("offline_checkpoint_type")
    if checkpoint_type is not None:
        return checkpoint_type == "decoder_encoder"
    return all(
        key in checkpoint
        for key in ("ppo_z_params", "q1_params", "q2_params", "value_params")
    )


def restore_decoder(
    decoder: DecoderFMState, checkpoint: dict[str, Any]
) -> DecoderFMState:
    params = checkpoint.get("params", checkpoint.get("fm_params"))
    obs_stats = checkpoint.get("obs_stats", checkpoint.get("fm_obs_stats"))
    if params is None or obs_stats is None:
        raise KeyError("Checkpoint is missing decoder params or observation stats.")
    with jdc.copy_and_mutate(decoder) as restored:
        restored.params = params
        restored.obs_stats = obs_stats
        if "decoder_opt_state" in checkpoint:
            restored.opt_state = checkpoint["decoder_opt_state"]
        if "decoder_prng" in checkpoint:
            restored.prng = checkpoint["decoder_prng"]
        if "decoder_steps" in checkpoint:
            restored.steps = checkpoint["decoder_steps"]
    return restored


def save_decoder_checkpoint(
    path: Path,
    config: ConfigView,
    decoder: DecoderFMState,
    decoder_epoch: int,
    global_step: int,
    best_params: PyTree,
    best_validation: float,
    stale_epochs: int,
    rng: np.random.Generator,
    key: Array,
) -> None:
    action_dim = int(decoder.params[-1][0].shape[-1])
    checkpoint = {
        "checkpoint_format": "gorl_offline_fm_decoder",
        "checkpoint_version": 2,
        "offline_checkpoint_type": "decoder",
        "training_phase": "decoder",
        # Keep the standard standalone decoder fields for interoperability.
        "params": decoder.params,
        "obs_stats": decoder.obs_stats,
        "config": decoder.config,
        "obs_dim": int(decoder.obs_stats.mean.shape[-1]),
        "action_dim": action_dim,
        "epoch": decoder_epoch,
        "decoder_epoch": decoder_epoch,
        "global_step": global_step,
        # Offline training state required for an exact continuation.
        "decoder_opt_state": decoder.opt_state,
        "decoder_prng": decoder.prng,
        "decoder_steps": decoder.steps,
        "decoder_best_params": best_params,
        "decoder_best_validation": best_validation,
        "decoder_stale_epochs": stale_epochs,
        "numpy_rng_state": rng.bit_generator.state,
        "jax_key": key,
        "offline_config": dict(config),
    }
    with open(path, "wb") as file:
        pickle.dump(checkpoint, file)


def save_compatible_checkpoint(
    path: Path,
    config: ConfigView,
    online_config: encoder_ppo.EncoderConfig,
    decoder: DecoderFMState,
    encoder_params: encoder_ppo.ActorCriticParams,
    encoder_obs_stats: Any,
    q1_params: PyTree,
    q2_params: PyTree,
    value_params: PyTree,
    decoder_epoch: int,
    encoder_iql_step: int,
    actor_opt_state: PyTree,
    critic_opt_state: PyTree,
    value_opt_state: PyTree,
    target_q1_params: PyTree,
    target_q2_params: PyTree,
    rng: np.random.Generator,
    key: Array,
) -> None:
    obs_dim = int(decoder.obs_stats.mean.shape[-1])
    action_dim = int(decoder.params[-1][0].shape[-1])
    checkpoint = {
        # Stable schema marker consumed by scripts/run_gorl_fm.py. Keep the
        # decoder fields below at the top level for train_encoder_ppo.py.
        "checkpoint_format": "gorl_fm_decoder",
        "checkpoint_version": 1,
        "offline_checkpoint_type": "decoder_encoder",
        "training_phase": "encoder",
        # Standalone FM schema loaded by train_encoder_ppo.py.
        "params": decoder.params,
        "obs_stats": decoder.obs_stats,
        "config": decoder.config,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "epoch": decoder_epoch,
        "decoder_epoch": decoder_epoch,
        "encoder_iql_step": encoder_iql_step,
        "is_frozen_offline": True,
        # Combined online encoder schema loaded by collect_data_fm.py.
        "ppo_z_params": encoder_params,
        "ppo_z_obs_stats": encoder_obs_stats,
        "env_name": config.env_name,
        "dataset_path": str(Path(config.dataset_path).expanduser().resolve()),
        "decoder_type": "fm",
        "z_dim": action_dim,
        "fm_params": decoder.params,
        "fm_obs_stats": decoder.obs_stats,
        "online_encoder_config": online_config,
        # Offline-only training state/metadata.
        "offline_config": dict(config),
        "q1_params": q1_params,
        "q2_params": q2_params,
        "value_params": value_params,
        "actor_opt_state": actor_opt_state,
        "critic_opt_state": critic_opt_state,
        "value_opt_state": value_opt_state,
        "target_q1_params": target_q1_params,
        "target_q2_params": target_q2_params,
        "numpy_rng_state": rng.bit_generator.state,
        "jax_key": key,
    }
    # collect_data_fm.py interprets "config" as EncoderConfig, while
    # train_encoder_ppo.py interprets it as DecoderFMConfig. One file cannot
    # place both types under the same key. The pipeline consumes this final
    # checkpoint as a decoder, so "config" stays DecoderFMConfig. A separate
    # encoder checkpoint is emitted below for collect-data/resume use.
    with open(path, "wb") as file:
        pickle.dump(checkpoint, file)


def save_encoder_checkpoint(
    path: Path,
    config: ConfigView,
    online_config: encoder_ppo.EncoderConfig,
    decoder: DecoderFMState,
    encoder_params: encoder_ppo.ActorCriticParams,
    encoder_obs_stats: Any,
) -> None:
    checkpoint = {
        "ppo_z_params": encoder_params,
        "ppo_z_obs_stats": encoder_obs_stats,
        "config": online_config,
        "env_name": config.env_name,
        "decoder_type": "fm",
        "iteration": 0,
        "reward": float("-inf"),
        "z_dim": int(decoder.params[-1][0].shape[-1]),
        "fm_params": decoder.params,
        "fm_obs_stats": decoder.obs_stats,
    }
    with open(path, "wb") as file:
        pickle.dump(checkpoint, file)


def main(config: ConfigView) -> None:
    validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    with open(output_dir / "config.json", "w") as file:
        json.dump(
            {**dict(config), "method": "frozen_decoder"},
            file,
            indent=2,
        )

    logger = WandbLogger(config)
    rng = np.random.default_rng(config.seed)
    try:
        resume_checkpoint, resume_path = load_checkpoint(config.checkpoint_path)
        env, environment_id = make_dataset_environment(config)
        buffer = load_replay_buffer(config, env)
        config.env_name = environment_id
        obs_dim = buffer.observations.shape[-1]
        action_dim = buffer.actions.shape[-1]
        if env.observation_size != obs_dim:
            raise ValueError(
                f"Dataset observation dim {obs_dim} does not match "
                f"{environment_id} robomimic observation size "
                f"{env.observation_size}."
            )
        if env.action_size != action_dim:
            raise ValueError(
                f"Dataset action dim {action_dim} does not match "
                f"{environment_id} robomimic action size {env.action_size}."
            )
        episode_length = int(config.episode_length)
        online_config = make_encoder_config(
            config, action_dim, episode_length
        )
        with open(output_dir / "config.json", "w") as file:
            json.dump(
                {
                    **dict(config),
                    "method": "frozen_decoder",
                    "resolved_environment": environment_id,
                },
                file,
                indent=2,
            )

        key = jax.random.key(config.seed)
        key, decoder_key, actor_key, value_key, q1_key, q2_key = (
            jax.random.split(key, 6)
        )
        decoder_config = DecoderFMConfig(
            flow_steps=config.flow_steps,
            timestep_embed_dim=config.timestep_embed_dim,
            hidden_dims=(config.decoder_hidden_size,)
            * config.decoder_num_layers,
            policy_output_scale=config.fm_policy_output_scale,
            learning_rate=config.decoder_learning_rate,
            batch_size=config.decoder_batch_size,
            num_epochs=config.decoder_max_epochs,
            n_samples_per_action=config.n_fm_samples_per_action,
            normalize_observations=config.fm_normalize_observations,
            sde_sigma=config.fm_sde_sigma,
            feather_std=config.fm_feather_std,
        )
        decoder = DecoderFMState.init(
            decoder_key, obs_dim, action_dim, decoder_config
        )
        with jdc.copy_and_mutate(decoder) as decoder:
            decoder.obs_stats = decoder.obs_stats.update(
                jnp.asarray(buffer.observations)
            )
        if resume_checkpoint is not None:
            decoder = restore_decoder(decoder, resume_checkpoint)

        # Keep the same policy/value layer layouts used by EncoderState.init,
        # sized from the matching robomimic environment.
        actor_params = networks.mlp_init(
            actor_key, (obs_dim, 32, 32, 32, 32, action_dim * 2)
        )
        value_params = networks.mlp_init(
            value_key, (obs_dim, 256, 256, 256, 256, 256, 1)
        )
        encoder_obs_stats = math_utils.RunningStats.init((obs_dim,)).update(
            jnp.asarray(buffer.observations)
        )
        obs_mean = np.asarray(encoder_obs_stats.mean)
        obs_std = np.asarray(encoder_obs_stats.std)
        normalized_observations = (
            buffer.observations - obs_mean
        ) / (obs_std + 1e-8)
        normalized_next_observations = (
            buffer.next_observations - obs_mean
        ) / (obs_std + 1e-8)

        q_dims = (
            obs_dim + action_dim,
            *((config.q_hidden_size,) * config.q_hidden_layers),
            1,
        )
        q1_params = networks.mlp_init(q1_key, q_dims)
        q2_params = networks.mlp_init(q2_key, q_dims)
        resume_encoder = (
            resume_checkpoint is not None
            and is_combined_checkpoint(resume_checkpoint)
        )
        start_iql_step = 0
        if resume_encoder:
            encoder_params = resume_checkpoint["ppo_z_params"]
            actor_params = encoder_params.policy
            value_params = resume_checkpoint.get(
                "value_params", encoder_params.value
            )
            q1_params = resume_checkpoint["q1_params"]
            q2_params = resume_checkpoint["q2_params"]
            encoder_obs_stats = resume_checkpoint.get(
                "ppo_z_obs_stats", encoder_obs_stats
            )
            obs_mean = np.asarray(encoder_obs_stats.mean)
            obs_std = np.asarray(encoder_obs_stats.std)
            normalized_observations = (
                buffer.observations - obs_mean
            ) / (obs_std + 1e-8)
            normalized_next_observations = (
                buffer.next_observations - obs_mean
            ) / (obs_std + 1e-8)
            start_iql_step = checkpoint_iql_step(resume_checkpoint, resume_path)
            print(
                f"Resuming encoder training from IQL step {start_iql_step}: "
                f"{resume_path}"
            )
        train_indices, validation_indices = split_indices(
            len(buffer), config.decoder_validation_fraction, rng
        )
        comparison_indices = rng.choice(
            validation_indices,
            min(config.comparison_samples, len(validation_indices)),
            replace=False,
        )
        print(
            f"Loaded {len(buffer):,} transitions for {environment_id}. "
            "Training decoder once, then freezing it."
        )

        start_decoder_epoch = 0
        best_params = jax.tree.map(jnp.copy, decoder.params)
        best_validation = float("inf")
        stale_epochs = 0
        global_step = 0
        if resume_checkpoint is not None:
            start_decoder_epoch = checkpoint_decoder_epoch(
                resume_checkpoint, resume_path
            )
            global_step = int(resume_checkpoint.get("global_step", 0))
            if "numpy_rng_state" in resume_checkpoint:
                rng.bit_generator.state = resume_checkpoint["numpy_rng_state"]
            key = resume_checkpoint.get("jax_key", key)
            best_params = resume_checkpoint.get("decoder_best_params", best_params)
            best_validation = float(
                resume_checkpoint.get("decoder_best_validation", best_validation)
            )
            stale_epochs = int(
                resume_checkpoint.get("decoder_stale_epochs", stale_epochs)
            )
            if not resume_encoder:
                print(
                    f"Resuming decoder training from epoch {start_decoder_epoch}: "
                    f"{resume_path}"
                )
        previous_latents: np.ndarray | None = None
        completed_decoder_epochs = start_decoder_epoch
        decoder_target_epoch = (
            start_decoder_epoch if resume_encoder else config.decoder_max_epochs
        )
        for epoch in trange(
            start_decoder_epoch,
            decoder_target_epoch,
            desc="Decoder epochs",
        ):
            losses = []
            permutation = rng.permutation(train_indices)
            for start in range(0, len(permutation), config.decoder_batch_size):
                batch = permutation[start : start + config.decoder_batch_size]
                decoder, metrics = decoder.train_step(
                    jnp.asarray(buffer.observations[batch]),
                    jnp.asarray(buffer.actions[batch]),
                )
                losses.append(float(metrics["loss"]))
                global_step += 1
            validation_loss, key = decoder_validation_loss(
                decoder,
                buffer,
                validation_indices,
                config.decoder_eval_batches,
                key,
            )
            comparison, current_latents = decoder_metrics(
                decoder,
                buffer,
                comparison_indices,
                config.latent_inverse_steps,
            )
            comparison["latent/drift_mse"] = (
                0.0
                if previous_latents is None
                else float(np.mean(np.square(current_latents - previous_latents)))
            )
            previous_latents = current_latents
            record = {
                "method": "frozen_decoder",
                "phase": "decoder",
                "global_step": global_step,
                "decoder/epoch": epoch + 1,
                "decoder/train_cfm_loss": float(np.mean(losses)),
                "decoder/validation_cfm_loss": validation_loss,
                **comparison,
            }
            append_metrics(metrics_path, record)
            logger.log(
                {k: v for k, v in record.items() if isinstance(v, (int, float))},
                global_step,
            )
            if validation_loss < best_validation - config.decoder_min_delta:
                best_validation = validation_loss
                best_params = jax.tree.map(jnp.copy, decoder.params)
                stale_epochs = 0
            else:
                stale_epochs += 1
            completed_decoder_epochs = epoch + 1
            if (
                completed_decoder_epochs % config.decoder_checkpoint_interval == 0
            ):
                decoder_checkpoint_path = (
                    output_dir
                    / f"decoder_checkpoint_epoch_{completed_decoder_epochs:09d}.pkl"
                )
                save_decoder_checkpoint(
                    decoder_checkpoint_path,
                    config,
                    decoder,
                    completed_decoder_epochs,
                    global_step,
                    best_params,
                    best_validation,
                    stale_epochs,
                    rng,
                    key,
                )
                print(
                    "Saved decoder checkpoint at epoch "
                    f"{completed_decoder_epochs}: {decoder_checkpoint_path}"
                )
            if (
                epoch + 1 >= config.decoder_min_epochs
                and stale_epochs >= config.decoder_patience
            ):
                print(f"Decoder early-stopped at epoch {epoch + 1}.")
                break

        decoder_epoch = completed_decoder_epochs
        if not resume_encoder:
            final_decoder_path = output_dir / "decoder_checkpoint_final.pkl"
            save_decoder_checkpoint(
                final_decoder_path,
                config,
                decoder,
                decoder_epoch,
                global_step,
                best_params,
                best_validation,
                stale_epochs,
                rng,
                key,
            )
            print(f"Saved final decoder checkpoint: {final_decoder_path}")
            # The decoder-only checkpoint above keeps the latest params aligned
            # with its optimizer state so increasing fm_num_epochs can resume
            # training correctly. Encoder training freezes the best validation
            # params, matching the original behavior.
            with jdc.copy_and_mutate(decoder) as decoder:
                decoder.params = best_params
        if np.isfinite(best_validation):
            print(f"Frozen decoder validation CFM loss: {best_validation:.6f}")
        else:
            print("Frozen decoder restored from checkpoint.")
        latent_targets = build_latent_targets(
            decoder,
            buffer,
            config.decoder_batch_size,
            config.latent_inverse_steps,
        )
        frozen_metrics, _ = decoder_metrics(
            decoder,
            buffer,
            comparison_indices,
            config.latent_inverse_steps,
        )
        frozen_record = {
            "method": "frozen_decoder",
            "phase": "decoder_frozen",
            "global_step": global_step,
            "decoder/best_validation_cfm_loss": best_validation,
            **frozen_metrics,
        }
        append_metrics(metrics_path, frozen_record)
        logger.log(
            {
                k: v
                for k, v in frozen_record.items()
                if isinstance(v, (int, float))
            },
            global_step,
        )

        actor_optimizer = optax.adam(config.actor_learning_rate)
        critic_optimizer = optax.adam(config.critic_learning_rate)
        value_optimizer = optax.adam(config.value_learning_rate)
        actor_opt_state = actor_optimizer.init(actor_params)
        critic_opt_state = critic_optimizer.init((q1_params, q2_params))
        value_opt_state = value_optimizer.init(value_params)
        target_q1_params = jax.tree.map(jnp.copy, q1_params)
        target_q2_params = jax.tree.map(jnp.copy, q2_params)
        if resume_encoder and resume_checkpoint is not None:
            actor_opt_state = resume_checkpoint.get(
                "actor_opt_state", actor_opt_state
            )
            critic_opt_state = resume_checkpoint.get(
                "critic_opt_state", critic_opt_state
            )
            value_opt_state = resume_checkpoint.get(
                "value_opt_state", value_opt_state
            )
            target_q1_params = resume_checkpoint.get(
                "target_q1_params", target_q1_params
            )
            target_q2_params = resume_checkpoint.get(
                "target_q2_params", target_q2_params
            )
        iql_update = make_iql_update(
            config, actor_optimizer, critic_optimizer, value_optimizer
        )
        accumulators: dict[str, list[float]] = {}
        policy_indices = rng.choice(
            len(buffer),
            min(config.comparison_samples, len(buffer)),
            replace=False,
        )
        encoder_target_step = max(start_iql_step, config.encoder_iql_steps)
        for step in trange(start_iql_step, encoder_target_step, desc="IQL"):
            indices = rng.integers(0, len(buffer), size=config.batch_size)
            (
                actor_params,
                actor_opt_state,
                q1_params,
                q2_params,
                critic_opt_state,
                value_params,
                value_opt_state,
                target_q1_params,
                target_q2_params,
                metrics,
            ) = iql_update(
                actor_params,
                actor_opt_state,
                q1_params,
                q2_params,
                critic_opt_state,
                value_params,
                value_opt_state,
                target_q1_params,
                target_q2_params,
                jnp.asarray(normalized_observations[indices]),
                jnp.asarray(buffer.actions[indices]),
                jnp.asarray(buffer.rewards[indices]),
                jnp.asarray(normalized_next_observations[indices]),
                jnp.asarray(buffer.masks[indices]),
                jnp.asarray(latent_targets[indices]),
            )
            for name, value in metrics.items():
                accumulators.setdefault(name, []).append(float(value))
            if (
                (step + 1) % config.log_interval == 0
                or step + 1 == encoder_target_step
            ):
                record = {
                    "method": "frozen_decoder",
                    "phase": "encoder",
                    "global_step": global_step + step + 1,
                    "encoder/iql_step": step + 1,
                    **{
                        f"iql/{name}": float(np.mean(values))
                        for name, values in accumulators.items()
                    },
                    **policy_metrics(
                        actor_params,
                        q1_params,
                        q2_params,
                        value_params,
                        decoder,
                        buffer,
                        normalized_observations,
                        policy_indices,
                        latent_targets,
                    ),
                }
                append_metrics(metrics_path, record)
                logger.log(
                    {
                        k: v
                        for k, v in record.items()
                        if isinstance(v, (int, float))
                    },
                    global_step + step + 1,
                )
                print(f"  step {step + 1}: {record}")
                accumulators.clear()

            completed_iql_steps = step + 1
            if completed_iql_steps % config.checkpoint_interval == 0:
                periodic_encoder_params = encoder_ppo.ActorCriticParams(
                    policy=actor_params, value=value_params
                )
                periodic_decoder_path = (
                    output_dir
                    / f"checkpoint_step_{completed_iql_steps:09d}.pkl"
                )
                periodic_encoder_path = (
                    output_dir
                    / f"encoder_checkpoint_step_{completed_iql_steps:09d}.pkl"
                )
                save_compatible_checkpoint(
                    periodic_decoder_path,
                    config,
                    online_config,
                    decoder,
                    periodic_encoder_params,
                    encoder_obs_stats,
                    q1_params,
                    q2_params,
                    value_params,
                    decoder_epoch,
                    completed_iql_steps,
                    actor_opt_state,
                    critic_opt_state,
                    value_opt_state,
                    target_q1_params,
                    target_q2_params,
                    rng,
                    key,
                )
                save_encoder_checkpoint(
                    periodic_encoder_path,
                    config,
                    online_config,
                    decoder,
                    periodic_encoder_params,
                    encoder_obs_stats,
                )
                print(
                    "Saved periodic checkpoints at IQL step "
                    f"{completed_iql_steps}: {periodic_decoder_path}, "
                    f"{periodic_encoder_path}"
                )

        trained_encoder_params = encoder_ppo.ActorCriticParams(
            policy=actor_params, value=value_params
        )
        decoder_path = output_dir / "checkpoint_final.pkl"
        encoder_path = output_dir / "encoder_checkpoint_final.pkl"
        final_iql_step = encoder_target_step
        save_compatible_checkpoint(
            decoder_path,
            config,
            online_config,
            decoder,
            trained_encoder_params,
            encoder_obs_stats,
            q1_params,
            q2_params,
            value_params,
            decoder_epoch,
            final_iql_step,
            actor_opt_state,
            critic_opt_state,
            value_opt_state,
            target_q1_params,
            target_q2_params,
            rng,
            key,
        )
        save_encoder_checkpoint(
            encoder_path,
            config,
            online_config,
            decoder,
            trained_encoder_params,
            encoder_obs_stats,
        )
        print(f"Saved online-compatible decoder checkpoint: {decoder_path}")
        print(f"Saved online-compatible encoder checkpoint: {encoder_path}")
    finally:
        logger.finish()


if __name__ == "__main__":
    main(build_config(tyro.cli(TrainingConfig)))
