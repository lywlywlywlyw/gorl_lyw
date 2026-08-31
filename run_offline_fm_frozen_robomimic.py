"""Frozen-decoder offline training for subsequent online RLPD training.

The method is the same two-stage procedure as ``run_offline_fm_frozen.py``:

1. Train a conditional flow-matching decoder to convergence and freeze it.
2. Invert dataset actions through the frozen decoder and train the latent
   encoder with IQL advantage-weighted behavior cloning.

This file is deliberately self-contained with respect to the old offline
scripts. It trains only the frozen FM decoder and IQL latent encoder needed to
initialize ``scripts/run_rlpd_fm.py``.

The final pickle contains:

* top-level ``params/obs_stats/config/obs_dim/action_dim`` FM decoder fields;
* ``iql_z_*`` fields for actor warm-start;
* ``rlpd_z_*`` fields for full RLPD actor/critic warm-start.

The combined checkpoints are marked as offline RLPD initialization artifacts
and can be passed directly to ``scripts/run_rlpd_fm.py`` via
``--use-offline-checkpoint --offline-checkpoint-path ...``.

Example:
    python run_offline_fm_frozen_robomimic.py

All environment, frozen-FM, IQL, checkpoint-metadata, and logging parameters
are sourced from ``envs.robomimic.offline_config``.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow JAX to grow its GPU allocation instead of reserving most VRAM upfront.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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
from envs.robomimic.online_config.encoder_configs.rlpd_config import RLPDConfig
from flow_policy import encoder_rlpd, math_utils, networks
from flow_policy.decoder_fm import DecoderFMConfig, DecoderFMState
from flow_policy.decoder_1step_fm_residualMLP import Decoder1StepFMConfig, Decoder1StepFMState


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
    config = ConfigView(
        training_config.to_dict() | EnvConfig().to_dict() | RLPDConfig().to_dict()
    )
    # Internal aliases keep the implementation names aligned with the generic
    # reference script while all values remain owned by offline_config.
    if config["decoder_type"] == "flow_matching":
        prefix = "fm"
    elif config["decoder_type"] == "meanflow":
        prefix = "meanflow"
    else:
        raise ValueError("decoder_type must be 'flow_matching' or 'meanflow'.")
    aliases = {
        "max_samples": "max_samples", "decoder_learning_rate": "learning_rate",
        "decoder_batch_size": "batch_size", "decoder_max_epochs": "num_epochs",
        "decoder_checkpoint_interval": "checkpoint_interval", "decoder_validation_fraction": "validation_split",
        "timestep_embed_dim": "timestep_embed_dim", "decoder_min_epochs": "min_epochs",
        "decoder_patience": "patience", "decoder_min_delta": "min_delta",
        "decoder_eval_batches": "eval_batches",
    }
    for target, suffix in aliases.items(): config[target] = config[f"{prefix}_{suffix}"]
    if prefix == "fm":
        config.update(decoder_hidden_size=config["fm_hidden_size"], decoder_num_layers=config["fm_num_layers"], flow_steps=config["fm_flow_steps"], latent_inverse_steps=config["fm_latent_inverse_steps"], n_fm_samples_per_action=config["fm_n_samples_per_action"])
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
            tags=["frozen_decoder", "rlpd_warm_start"],
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
        config.comparison_samples,
        config.checkpoint_interval,
        config.decoder_checkpoint_interval,
        config.validation_interval,
        config.validation_batches,
        config.early_stopping_min_steps,
        config.early_stopping_patience,
    ) < 1:
        raise ValueError("Batch sizes and step/sample counts must be positive.")
    if config.decoder_type == "flow_matching" and config.latent_inverse_steps < 1:
        raise ValueError("latent_inverse_steps must be positive for Flow Matching.")
    if config.max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive.")
    if config.early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative.")
    if config.early_stopping_actor_nll_weight < 0.0:
        raise ValueError(
            "early_stopping_actor_nll_weight must be non-negative."
        )
    if config.wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline, or disabled.")


def make_dataset_environment(
    config: ConfigView,
) -> tuple[RobomimicEnv, str]:
    """Create robomimic exactly as in ``scripts/run_gorl_fm.py``."""
    environment = RobomimicEnv(dataset_path=config.dataset_path, reward_shaping=config.dense_reward)
    return environment, config.env_name


def make_rlpd_encoder_config(
    config: ConfigView,
    action_dim: int,
    episode_length: int,
) -> encoder_rlpd.EncoderConfig:
    """Build the exact encoder architecture used by online RLPD training."""
    return encoder_rlpd.EncoderConfig(
        learning_rate=config.rlpd_actor_learning_rate,
        critic_learning_rate=config.rlpd_critic_learning_rate,
        temperature_learning_rate=config.rlpd_temperature_learning_rate,
        learn_temperature=config.rlpd_learn_temperature,
        discounting=config.rlpd_discounting,
        episode_length=episode_length,
        normalize_observations=config.rlpd_normalize_observations,
        num_envs=config.num_envs,
        z_dim=action_dim,
        hidden_size=config.rlpd_hidden_size,
        hidden_layers=config.rlpd_hidden_layers,
        critic_ensemble_size=config.rlpd_critic_ensemble_size,
        critic_subsample_size=config.rlpd_critic_subsample_size,
        target_update_rate=config.rlpd_target_update_rate,
        initial_temperature=config.rlpd_initial_temperature,
        target_entropy=config.rlpd_target_entropy,
        backup_entropy=config.rlpd_backup_entropy,
        reward_scaling=config.rlpd_reward_scaling,
        reward_bias=config.rlpd_reward_bias,
        max_grad_norm=config.rlpd_max_grad_norm,
        latent_kl_weight=config.rlpd_latent_kl_weight,
        latent_kl_threshold=config.rlpd_latent_kl_threshold,
        latent_kl_dual_learning_rate=config.rlpd_latent_kl_dual_learning_rate,
        latent_prior_support_radius=config.rlpd_latent_prior_support_radius,
        latent_policy_support_stddevs=config.rlpd_latent_policy_support_stddevs,
        actor_mean_bound=config.rlpd_actor_mean_bound,
        policy_update_period=config.rlpd_policy_update_period,
        apply_tanh_in_rollout=config.rlpd_apply_tanh_in_rollout,
    )


def split_indices(
    size: int, validation_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if size < 2:
        raise ValueError("At least two transitions are required.")
    permutation = rng.permutation(size)
    validation_size = min(size - 1, max(1, int(size * validation_fraction)))
    return permutation[validation_size:], permutation[:validation_size]

def forward_fm_batch(
    decoder: Any, observations: Array, latents: Array
) -> Array:
    if isinstance(decoder, Decoder1StepFMState):
        return decoder.sample_action_from_z(observations, latents, jax.random.PRNGKey(0), deterministic=True)
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
    return jnp.clip(actions, -1.0, 1.0)


def build_latent_targets(
    decoder: Any,
    buffer: ReplayBuffer,
    batch_size: int,
    inverse_steps: int | None,
) -> np.ndarray:
    if isinstance(decoder, Decoder1StepFMState):
        invert = jax.jit(decoder.inverse_fm_batch)
    else:
        if inverse_steps is None:
            raise ValueError("Flow Matching inversion requires inverse_steps.")
        invert = jax.jit(
            lambda obs, act: decoder.inverse_fm_batch(obs, act, inverse_steps)
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
    decoder: Any,
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
        key, eps_key, time_key = jax.random.split(key, 3)
        if isinstance(decoder, Decoder1StepFMState):
            eps = jax.random.normal(eps_key, actions.shape)
            times, starts = decoder.sample_t_r(time_key, len(batch))
            losses.append(float(decoder.compute_meanflow_loss(decoder.steps, decoder._normalize_obs(obs), actions, eps, times, starts)[0]))
        else:
            obs_norm = (obs - decoder.obs_stats.mean) / (decoder.obs_stats.std + 1e-8)
            eps = jax.random.normal(eps_key, (len(batch), decoder.config.n_samples_per_action, actions.shape[-1]))
            times = jax.random.uniform(time_key, (len(batch), decoder.config.n_samples_per_action, 1))
            losses.append(float(jnp.mean(decoder.compute_cfm_loss(obs_norm, actions, eps, times))))
    return float(np.mean(losses)), key


def decoder_metrics(
    decoder: Any,
    buffer: ReplayBuffer,
    indices: np.ndarray,
    inverse_steps: int | None,
) -> tuple[dict[str, float], np.ndarray]:
    obs = jnp.asarray(buffer.observations[indices])
    actions = jnp.asarray(buffer.actions[indices])
    if isinstance(decoder, Decoder1StepFMState):
        latents = jax.jit(decoder.inverse_fm_batch)(obs, actions)
    else:
        if inverse_steps is None:
            raise ValueError("Flow Matching inversion requires inverse_steps.")
        latents = jax.jit(
            lambda o, a: decoder.inverse_fm_batch(o, a, inverse_steps)
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
        # The encoder chooses decoder latents, so IQL must estimate Q(s, z).
        # Keeping environment actions here would train a shape-compatible but
        # semantically incompatible critic for online latent-space RLPD.
        target_q = jnp.minimum(
            networks.q_mlp_fwd(target_q1_params, obs, latent_actions),
            networks.q_mlp_fwd(target_q2_params, obs, latent_actions),
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
            distribution = networks.gaussian_policy_fwd(
                params, obs, mean_bound=config.rlpd_actor_mean_bound
            )
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
            q1_value = networks.q_mlp_fwd(q1, obs, latent_actions)
            q2_value = networks.q_mlp_fwd(q2, obs, latent_actions)
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

def iql_validation_losses(
    config: ConfigView,
    actor_params: PyTree,
    q1_params: PyTree,
    q2_params: PyTree,
    value_params: PyTree,
    target_q1_params: PyTree,
    target_q2_params: PyTree,
    normalized_observations: np.ndarray,
    normalized_next_observations: np.ndarray,
    buffer: ReplayBuffer,
    latent_targets: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    """Compute deterministic held-out Bellman TD and expectile value losses."""
    obs = jnp.asarray(normalized_observations[indices])
    next_obs = jnp.asarray(normalized_next_observations[indices])
    latents = jnp.asarray(latent_targets[indices])
    rewards = jnp.asarray(buffer.rewards[indices])
    masks = jnp.asarray(buffer.masks[indices])
    target_q = jnp.minimum(
        networks.q_mlp_fwd(target_q1_params, obs, latents),
        networks.q_mlp_fwd(target_q2_params, obs, latents),
    )
    distribution = networks.gaussian_policy_fwd(
        actor_params, obs, mean_bound=config.rlpd_actor_mean_bound
    )
    actor_nll = -jnp.mean(jnp.sum(distribution.log_prob(latents), axis=-1))
    value = networks.value_mlp_fwd(value_params, obs)
    value_loss = jnp.mean(expectile_loss(target_q - value, config.expectile))
    next_value = networks.value_mlp_fwd(value_params, next_obs)
    bellman_target = rewards + config.discount * masks * next_value
    q1 = networks.q_mlp_fwd(q1_params, obs, latents)
    q2 = networks.q_mlp_fwd(q2_params, obs, latents)
    td_loss = jnp.mean(
        jnp.square(q1 - bellman_target) + jnp.square(q2 - bellman_target)
    )
    q_scale = jnp.maximum(jnp.mean(jnp.abs(bellman_target)), 1.0)
    value_scale = jnp.maximum(jnp.mean(jnp.abs(target_q)), 1.0)
    relative_td_rmse = jnp.sqrt(td_loss / 2.0) / q_scale
    relative_value_rmse = jnp.sqrt(value_loss) / value_scale
    actor_nll_per_dim = actor_nll / latents.shape[-1]
    validation_score = (
        relative_td_rmse
        + relative_value_rmse
        + config.early_stopping_actor_nll_weight * actor_nll_per_dim
    )
    return {
        "validation_td_loss": float(td_loss),
        "validation_value_loss": float(value_loss),
        "validation_loss": float(td_loss + value_loss),
        "validation_q_scale": float(q_scale),
        "validation_value_scale": float(value_scale),
        "validation_relative_td_rmse": float(relative_td_rmse),
        "validation_relative_value_rmse": float(relative_value_rmse),
        "validation_actor_nll": float(actor_nll),
        "validation_actor_nll_per_dim": float(actor_nll_per_dim),
        "validation_score": float(validation_score),
    }


def compatible_opt_state(
    checkpoint: dict[str, Any], name: str, default: PyTree
) -> PyTree:
    """Restore optimizer state only when its transform structure matches."""
    candidate = checkpoint.get(name)
    if candidate is None:
        return default
    if (
        jax.tree_util.tree_structure(candidate)
        != jax.tree_util.tree_structure(default)
    ):
        print(f"Ignoring incompatible legacy optimizer state: {name}.")
        return default
    return candidate


def policy_metrics(
    actor_params: PyTree,
    q1_params: PyTree,
    q2_params: PyTree,
    value_params: PyTree,
    decoder: Any,
    buffer: ReplayBuffer,
    normalized_observations: np.ndarray,
    indices: np.ndarray,
    latent_targets: np.ndarray,
    actor_mean_bound: float,
) -> dict[str, float]:
    obs_norm = jnp.asarray(normalized_observations[indices])
    obs_raw = jnp.asarray(buffer.observations[indices])
    data_actions = jnp.asarray(buffer.actions[indices])
    targets = jnp.asarray(latent_targets[indices])
    distribution = networks.gaussian_policy_fwd(
        actor_params, obs_norm, mean_bound=actor_mean_bound
    )
    policy_z = distribution.loc
    policy_actions = jax.jit(
        lambda o, z: forward_fm_batch(decoder, o, z)
    )(obs_raw, policy_z)
    q_policy = jnp.minimum(
        networks.q_mlp_fwd(q1_params, obs_norm, policy_z),
        networks.q_mlp_fwd(q2_params, obs_norm, policy_z),
    )
    q_data = jnp.minimum(
        networks.q_mlp_fwd(q1_params, obs_norm, targets),
        networks.q_mlp_fwd(q2_params, obs_norm, targets),
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
        for key in (
            "iql_z_actor_params",
            "q1_params",
            "q2_params",
            "value_params",
        )
    )


def restore_decoder(
    decoder: Any, checkpoint: dict[str, Any]
) -> DecoderFMState:
    params = checkpoint.get("params")
    obs_stats = checkpoint.get("obs_stats")
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
    decoder: Any,
    decoder_epoch: int,
    global_step: int,
    best_params: PyTree,
    best_validation: float,
    stale_epochs: int,
    rng: np.random.Generator,
    key: Array,
) -> None:
    action_dim = int(decoder.action_dim if hasattr(decoder, "action_dim") else decoder.params[-1][0].shape[-1])
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


def save_offline_checkpoint(
    path: Path,
    config: ConfigView,
    rlpd_config: encoder_rlpd.EncoderConfig,
    decoder: Any,
    actor_params: PyTree,
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
    action_dim = int(decoder.action_dim if hasattr(decoder, "action_dim") else decoder.params[-1][0].shape[-1])
    critic_params = tuple(
        q1_params if member % 2 == 0 else q2_params
        for member in range(rlpd_config.critic_ensemble_size)
    )
    target_critic_params = tuple(
        target_q1_params if member % 2 == 0 else target_q2_params
        for member in range(rlpd_config.critic_ensemble_size)
    )
    checkpoint = {
        "checkpoint_format": "gorl_offline_fm_rlpd",
        "checkpoint_version": 2,
        "offline_checkpoint_type": "decoder_encoder",
        "training_phase": "encoder",
        # Frozen FM decoder loaded by the RLPD training component.
        "params": decoder.params,
        "obs_stats": decoder.obs_stats,
        "config": decoder.config,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "epoch": decoder_epoch,
        "decoder_epoch": decoder_epoch,
        "encoder_iql_step": encoder_iql_step,
        "is_frozen_offline": True,
        "env_name": config.env_name,
        "dataset_path": str(Path(config.dataset_path).expanduser().resolve()),
        "decoder_type": config.decoder_type,
        "z_dim": action_dim,
        "fm_params": decoder.params,
        "fm_obs_stats": decoder.obs_stats,
        "rlpd_encoder_config": rlpd_config,
        # Stage-0 RLPD warm start used when stage_init_before_training=True.
        "iql_z_actor_params": actor_params,
        "iql_z_obs_stats": encoder_obs_stats,
        # Full RLPD-compatible state used when stage_init_before_training=False.
        # IQL trains two Q networks; expand them alternately to the configured
        # RLPD ensemble so every online critic starts from an offline-trained Q.
        "rlpd_z_actor_params": actor_params,
        "rlpd_z_critic_params": critic_params,
        "rlpd_z_target_critic_params": target_critic_params,
        "rlpd_temperature_parameterization": "softplus_raw",
        "rlpd_z_log_temperature": jnp.log(
            jnp.expm1(jnp.asarray(rlpd_config.initial_temperature))
        ),
        "rlpd_z_latent_kl_multiplier": jnp.asarray(
            rlpd_config.latent_kl_weight
        ),
        "rlpd_z_obs_stats": encoder_obs_stats,
        # Offline-only training state/metadata.
        "decoder_type": config.decoder_type,
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
        rlpd_config = make_rlpd_encoder_config(
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
        if config.decoder_type == "meanflow":
            decoder_config = Decoder1StepFMConfig(
                timestep_embed_dim=config.meanflow_timestep_embed_dim,
                hidden_dim=config.meanflow_hidden_dim, num_res_blocks=config.meanflow_num_res_blocks,
                mlp_expansion=config.meanflow_mlp_expansion,
                policy_output_scale=config.meanflow_policy_output_scale, learning_rate=config.decoder_learning_rate,
                batch_size=config.decoder_batch_size,
                normalize_observations=config.meanflow_normalize_observations,
                normalization_mode="gaussian",
                flow_ratio=config.meanflow_flow_ratio,
                guidance_scale=config.meanflow_guidance_scale, use_dispersive=config.use_dispersive, dispersive_loss_weight=config.meanflow_dispersive_loss_weight,
                latent_kl_weight=config.rlpd_latent_kl_weight,
            )
            decoder = Decoder1StepFMState.init(decoder_key, obs_dim, action_dim, decoder_config)
            with jdc.copy_and_mutate(decoder) as decoder:
                decoder.obs_stats = decoder.obs_stats.update(jnp.asarray(buffer.observations))
        else:
            decoder_config = DecoderFMConfig(flow_steps=config.flow_steps, timestep_embed_dim=config.timestep_embed_dim,
                hidden_dims=(config.decoder_hidden_size,) * config.decoder_num_layers,
                policy_output_scale=config.fm_policy_output_scale, learning_rate=config.decoder_learning_rate,
                batch_size=config.decoder_batch_size, num_epochs=config.decoder_max_epochs,
                n_samples_per_action=config.n_fm_samples_per_action, normalize_observations=config.fm_normalize_observations,
                sde_sigma=config.fm_sde_sigma, feather_std=config.fm_feather_std)
            decoder = DecoderFMState.init(decoder_key, obs_dim, action_dim, decoder_config)
            with jdc.copy_and_mutate(decoder) as decoder:
                decoder.obs_stats = decoder.obs_stats.update(jnp.asarray(buffer.observations))
        if resume_checkpoint is not None:
            decoder = restore_decoder(decoder, resume_checkpoint)

        # IQL keeps its original actor/value training objectives, but the actor
        # layout matches EncoderState.init in encoder_rlpd.py so its parameters
        # can initialize online RLPD without conversion.
        actor_params = networks.mlp_init(
            actor_key,
            (obs_dim,)
            + (rlpd_config.hidden_size,) * rlpd_config.hidden_layers
            + (action_dim * 2,),
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

        if (
            config.q_hidden_size != rlpd_config.hidden_size
            or config.q_hidden_layers != rlpd_config.hidden_layers
        ):
            raise ValueError(
                "Offline IQL Q architecture must match online RLPD: "
                f"IQL=({config.q_hidden_size}, {config.q_hidden_layers}), "
                f"RLPD=({rlpd_config.hidden_size}, {rlpd_config.hidden_layers})."
            )
        q_dims = (
            obs_dim + action_dim,
            *((rlpd_config.hidden_size,) * rlpd_config.hidden_layers),
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
            actor_params = resume_checkpoint["iql_z_actor_params"]
            value_params = resume_checkpoint["value_params"]
            q1_params = resume_checkpoint["q1_params"]
            q2_params = resume_checkpoint["q2_params"]
            encoder_obs_stats = resume_checkpoint.get(
                "iql_z_obs_stats", encoder_obs_stats
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
                if isinstance(decoder, Decoder1StepFMState):
                    decoder, metrics = decoder.train_step(
                        epoch, jnp.asarray(buffer.observations[batch]), jnp.asarray(buffer.actions[batch])
                    )
                else:
                    decoder, metrics = decoder.train_step(
                        jnp.asarray(buffer.observations[batch]),
                        # Train directly on dataset (s, a). Dataset actions are
                        # already the targets; never apply atanh/arctanh here.
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
                config.latent_inverse_steps
                if config.decoder_type == "flow_matching"
                else None,
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
            config.latent_inverse_steps
            if config.decoder_type == "flow_matching"
            else None,
        )
        frozen_metrics, _ = decoder_metrics(
            decoder,
            buffer,
            comparison_indices,
            config.latent_inverse_steps
            if config.decoder_type == "flow_matching"
            else None,
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

        actor_optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.actor_learning_rate),
        )
        critic_optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.critic_learning_rate),
        )
        value_optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.value_learning_rate),
        )
        actor_opt_state = actor_optimizer.init(actor_params)
        critic_opt_state = critic_optimizer.init((q1_params, q2_params))
        value_opt_state = value_optimizer.init(value_params)
        target_q1_params = jax.tree.map(jnp.copy, q1_params)
        target_q2_params = jax.tree.map(jnp.copy, q2_params)
        if resume_encoder and resume_checkpoint is not None:
            actor_opt_state = compatible_opt_state(
                resume_checkpoint, "actor_opt_state", actor_opt_state
            )
            critic_opt_state = compatible_opt_state(
                resume_checkpoint, "critic_opt_state", critic_opt_state
            )
            value_opt_state = compatible_opt_state(
                resume_checkpoint, "value_opt_state", value_opt_state
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
        validation_eval_indices = rng.choice(
            validation_indices,
            min(
                len(validation_indices),
                config.validation_batches * config.batch_size,
            ),
            replace=False,
        )
        latest_validation_metrics: dict[str, float] = {}
        best_validation_score = float("inf")
        best_iql_step = start_iql_step
        best_iql_state = None
        stale_validations = 0
        encoder_target_step = max(start_iql_step, config.encoder_iql_steps)
        completed_iql_steps = start_iql_step
        for step in trange(start_iql_step, encoder_target_step, desc="IQL"):
            indices = rng.choice(
                train_indices, size=config.batch_size, replace=True
            )
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
            completed_iql_steps = step + 1
            should_validate = (
                completed_iql_steps % config.validation_interval == 0
                or completed_iql_steps == encoder_target_step
            )
            should_stop = False
            if should_validate:
                latest_validation_metrics = iql_validation_losses(
                    config,
                    actor_params,
                    q1_params,
                    q2_params,
                    value_params,
                    target_q1_params,
                    target_q2_params,
                    normalized_observations,
                    normalized_next_observations,
                    buffer,
                    latent_targets,
                    validation_eval_indices,
                )
                if completed_iql_steps >= config.early_stopping_min_steps:
                    validation_score = latest_validation_metrics[
                        "validation_score"
                    ]
                    if validation_score < (
                        best_validation_score - config.early_stopping_min_delta
                    ):
                        best_validation_score = validation_score
                        best_iql_step = completed_iql_steps
                        best_iql_state = (
                            actor_params, actor_opt_state,
                            q1_params, q2_params, critic_opt_state,
                            value_params, value_opt_state,
                            target_q1_params, target_q2_params,
                        )
                        stale_validations = 0
                    else:
                        stale_validations += 1
                    should_stop = (
                        stale_validations >= config.early_stopping_patience
                    )
            if (
                (step + 1) % config.log_interval == 0
                or completed_iql_steps == encoder_target_step
                or should_validate
            ):
                record = {
                    "method": "frozen_decoder",
                    "phase": "encoder",
                    "global_step": global_step + step + 1,
                    "encoder/iql_step": step + 1,
                    "encoder/best_iql_step": best_iql_step,
                    "encoder/best_validation_score": best_validation_score,
                    "encoder/stale_validations": stale_validations,
                    **{
                        f"iql/{name}": value
                        for name, value in latest_validation_metrics.items()
                    },
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
                        rlpd_config.actor_mean_bound,
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

            if completed_iql_steps % config.checkpoint_interval == 0:
                periodic_decoder_path = (
                    output_dir
                    / f"checkpoint_step_{completed_iql_steps:09d}.pkl"
                )
                save_offline_checkpoint(
                    periodic_decoder_path,
                    config,
                    rlpd_config,
                    decoder,
                    actor_params,
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
                print(
                    "Saved periodic offline RLPD checkpoint at IQL step "
                    f"{completed_iql_steps}: {periodic_decoder_path}"
                )

            if should_stop:
                print(
                    f"IQL early-stopped at step {completed_iql_steps}; "
                    f"best held-out score {best_validation_score:.6f} "
                    f"at step {best_iql_step}."
                )
                break
        if best_iql_state is None:
            best_iql_step = completed_iql_steps
            best_iql_state = (
                actor_params, actor_opt_state,
                q1_params, q2_params, critic_opt_state,
                value_params, value_opt_state,
                target_q1_params, target_q2_params,
            )
        (
            actor_params,
            actor_opt_state,
            q1_params, q2_params, critic_opt_state,
            value_params, value_opt_state,
            target_q1_params, target_q2_params,
        ) = best_iql_state
        print(
            f"Restored best encoder from IQL step {best_iql_step}."
        )
        decoder_path = output_dir / "checkpoint_final.pkl"
        final_iql_step = best_iql_step
        save_offline_checkpoint(
            decoder_path,
            config,
            rlpd_config,
            decoder,
            actor_params,
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
        print(f"Saved final offline RLPD checkpoint: {decoder_path}")
    finally:
        logger.finish()


if __name__ == "__main__":
    main(build_config(tyro.cli(TrainingConfig)))
