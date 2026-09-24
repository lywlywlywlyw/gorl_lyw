"""Frozen-decoder offline training for subsequent online RLPD training.

The method is the same two-stage procedure as ``run_offline_fm_frozen.py``:

1. Train a conditional flow-matching decoder to convergence and freeze it.
2. Invert dataset actions through the frozen decoder and train the latent
   encoder with IQL advantage-weighted behavior cloning.
3. After IQL early stopping, finetune only the critics with an
   online-RLPD-compatible latent score-matching bridge.  A frozen copy of the
   early-stopped actor supplies the teacher score, and the live actor remains
   frozen at its early-stopped parameters.

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
    """Compose a robomimic or D4RL frozen-IQL configuration."""
    training_config = training_config or TrainingConfig()
    backend = getattr(training_config, "environment", "robomimic")
    if backend == "robomimic":
        from envs.robomimic.offline_config.env_config import EnvConfig
    elif backend == "d4rl":
        from envs.d4rl.offline_config.env_config import EnvConfig
    else:
        raise ValueError("environment must be 'robomimic' or 'd4rl'.")
    config = ConfigView(
        training_config.to_dict() | EnvConfig().to_dict() | RLPDConfig().to_dict()
    )
    config["environment"] = backend
    if getattr(training_config, "dataset_path", None):
        config["dataset_path"] = training_config.dataset_path
    if backend == "d4rl":
        from envs.d4rl.D4RLEnv import infer_env_name
        config["env_name"] = infer_env_name(config["dataset_path"])
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
    config: ConfigView, environment: Any
) -> ReplayBuffer:
    """Load transitions from the selected robomimic or D4RL dataset."""
    if config.get("environment", "robomimic") == "d4rl":
        dataset = environment.get_dataset()
        observations = np.asarray(dataset["observations"], dtype=np.float32)
        actions = np.asarray(dataset["actions"], dtype=np.float32)
        rewards = np.asarray(dataset.get("rewards", np.zeros(len(actions))), dtype=np.float32).reshape(-1)
        next_observations = np.asarray(dataset["next_observations"], dtype=np.float32)
        if "dones" not in dataset:
            raise KeyError("D4RL training data must contain dones; run the preprocessing script first.")
        done = np.asarray(dataset["dones"], dtype=np.float32).reshape(-1)
        if not (len(observations) == len(actions) == len(rewards) == len(next_observations) == len(done)):
            raise ValueError("D4RL dataset transition arrays have inconsistent lengths.")
        buffer = ReplayBuffer(observations, actions, rewards, next_observations, 1.0 - done)
        if config.max_samples is not None and len(buffer) > config.max_samples:
            rng = np.random.default_rng(config.seed)
            indices = rng.choice(len(buffer), config.max_samples, replace=False)
            buffer = ReplayBuffer(*(array[indices] for array in (buffer.observations, buffer.actions, buffer.rewards, buffer.next_observations, buffer.masks)))
        return buffer
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
    if config.early_stopping_actor_nll_weight < 0.0:
        raise ValueError(
            "early_stopping_actor_nll_weight must be non-negative."
        )
    if config.encoder_iql_prior_kl_weight < 0.0:
        raise ValueError(
            "encoder_iql_prior_kl_weight must be non-negative."
        )
    if config.encoder_alignment_steps < 0:
        raise ValueError("encoder_alignment_steps must be non-negative.")
    if config.encoder_score_matching_weight < 0.0:
        raise ValueError(
            "encoder_score_matching_weight must be non-negative."
        )
    if config.wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline, or disabled.")


def make_dataset_environment(config: ConfigView) -> tuple[Any, str]:
    if config.get("environment", "robomimic") == "d4rl":
        from envs.d4rl.D4RLEnv import D4RLEnv
        return D4RLEnv(dataset_path=config.dataset_path, reward_shaping=config.dense_reward), config.env_name
    from envs.robomimic.RobomimicEnv import RobomimicEnv
    return RobomimicEnv(dataset_path=config.dataset_path, reward_shaping=config.dense_reward), config.env_name


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
    observations: Array,
    actions: Array,
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
        0, observations.shape[0], batch_size, desc="Invert actions", leave=False
    ):
        end = min(start + batch_size, observations.shape[0])
        chunks.append(
            np.asarray(
                invert(
                    observations[start:end], actions[start:end],
                )
            )
        )
    return np.concatenate(chunks)


def make_decoder_eval_functions(decoder: Any, inverse_steps: int | None):
    """Create decoder evaluation kernels once; parameters remain dynamic."""
    if isinstance(decoder, Decoder1StepFMState):
        @jax.jit
        def inverse(state, obs, actions):
            return state.inverse_fm_batch(obs, actions)

        @jax.jit
        def validation(state, obs, actions, key):
            key, eps_key, time_key = jax.random.split(key, 3)
            eps = jax.random.normal(eps_key, actions.shape)
            times, starts = state.sample_t_r(time_key, actions.shape[0])
            loss = state.compute_meanflow_loss(
                state.steps, state._normalize_obs(obs), actions,
                eps, times, starts
            )[0]
            return loss, key
    else:
        if inverse_steps is None:
            raise ValueError("Flow Matching inversion requires inverse_steps.")

        @jax.jit
        def inverse(state, obs, actions):
            return state.inverse_fm_batch(obs, actions, inverse_steps)

        @jax.jit
        def validation(state, obs, actions, key):
            key, eps_key, time_key = jax.random.split(key, 3)
            obs_norm = (obs - state.obs_stats.mean) / (state.obs_stats.std + 1e-8)
            eps = jax.random.normal(
                eps_key,
                (actions.shape[0], state.config.n_samples_per_action, actions.shape[-1]),
            )
            times = jax.random.uniform(
                time_key, (actions.shape[0], state.config.n_samples_per_action, 1)
            )
            return jnp.mean(state.compute_cfm_loss(obs_norm, actions, eps, times)), key

    @jax.jit
    def metrics(state, obs, actions):
        latents = inverse(state, obs, actions)
        reconstructed = forward_fm_batch(state, obs, latents)
        generated_latents = jax.random.normal(
            jax.random.PRNGKey(0), actions.shape
        )
        generated_actions = forward_fm_batch(state, obs, generated_latents)
        recovered_latents = inverse(state, obs, generated_actions)
        latent_std = jnp.std(latents, axis=0)
        values = jnp.asarray([
            jnp.mean(jnp.square(reconstructed - actions)),
            jnp.mean(jnp.abs(reconstructed - actions)),
            jnp.mean(jnp.square(recovered_latents - generated_latents)),
            jnp.sqrt(jnp.mean(jnp.square(recovered_latents - generated_latents)) + 1e-8),
            jnp.mean(jnp.abs(jnp.mean(latents, axis=0))),
            jnp.mean(latent_std),
            jnp.mean(jnp.abs(latent_std - 1.0)),
            jnp.mean(jnp.linalg.norm(latents, axis=-1)),
            jnp.max(jnp.abs(latents)),
        ])
        return values, latents

    return inverse, validation, metrics


def decoder_validation_loss(
    decoder: Any,
    observations: Array,
    actions: Array,
    indices: np.ndarray,
    max_batches: int,
    key: Array,
    validation_fn,
) -> tuple[float, Array]:
    losses = []
    for batch_number, start in enumerate(
        range(0, len(indices), decoder.config.batch_size)
    ):
        if batch_number >= max_batches:
            break
        batch = indices[start : start + decoder.config.batch_size]
        obs = observations[jnp.asarray(batch)]
        batch_actions = actions[jnp.asarray(batch)]
        loss, key = validation_fn(decoder, obs, batch_actions, key)
        losses.append(loss)
    return float(jax.device_get(jnp.mean(jnp.stack(losses)))), key


def decoder_metrics(
    decoder: Any,
    observations: Array,
    actions: Array,
    indices: np.ndarray,
    metrics_fn,
) -> tuple[dict[str, float], np.ndarray]:
    obs = observations[jnp.asarray(indices)]
    batch_actions = actions[jnp.asarray(indices)]
    values, latents = metrics_fn(decoder, obs, batch_actions)
    values, latents = jax.device_get((values, latents))
    names = (
        "decoder/inverse_action_mse",
        "decoder/inverse_action_mae",
        "decoder/cycle_z_mse",
        "decoder/cycle_z_rmse",
        "latent/mean_abs",
        "latent/std_mean",
        "latent/std_error",
        "latent/norm_mean",
        "latent/max_abs",
    )
    return dict(zip(names, np.asarray(values).tolist())), np.asarray(latents)


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
        critic_params,
        critic_opt_state,
        value_params,
        value_opt_state,
        target_critic_params,
        obs,
        actions,
        rewards,
        next_obs,
        masks,
        latent_actions,
        critic_subset_key,
    ):
        target_qs = networks.q_ensemble_values(target_critic_params, obs, latent_actions)
        subset_size = min(
            int(config.rlpd_critic_subsample_size),
            len(target_critic_params.backbones),
        )
        subset = jax.random.randint(
            critic_subset_key,
            shape=(subset_size,),
            minval=0,
            maxval=len(target_critic_params.backbones),
        )
        target_q = jnp.min(target_qs[subset], axis=0)

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
            mean = distribution.loc
            std = distribution.scale
            prior_kl = 0.5 * jnp.sum(
                jnp.square(mean) + jnp.square(std) - 1.0 - 2.0 * jnp.log(std),
                axis=-1,
            )
            actor_loss = -jnp.mean(advantage_weight * log_prob)
            actor_loss += config.encoder_iql_prior_kl_weight * jnp.mean(prior_kl)
            return actor_loss, jnp.mean(prior_kl)

        (actor_loss, latent_prior_kl), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(actor_params)
        actor_updates, actor_opt_state = actor_optimizer.update(
            actor_grads, actor_opt_state, actor_params
        )
        actor_params = optax.apply_updates(actor_params, actor_updates)

        next_value = jax.lax.stop_gradient(
            networks.value_mlp_fwd(value_params, next_obs)
        )
        bellman_target = rewards + config.discount * masks * next_value

        def critic_loss_fn(params):
            predicted = networks.q_ensemble_values(params, obs, latent_actions)
            loss = jnp.mean(jnp.square(predicted - bellman_target[None, :]))
            return loss, predicted

        (critic_loss, predicted_qs), critic_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(critic_params)
        critic_updates, critic_opt_state = critic_optimizer.update(
            critic_grads, critic_opt_state, critic_params
        )
        critic_params = optax.apply_updates(critic_params, critic_updates)
        target_critic_params = polyak_update(
            critic_params, target_critic_params, config.target_update_rate
        )
        return (
            actor_params,
            actor_opt_state,
            critic_params,
            critic_opt_state,
            value_params,
            value_opt_state,
            target_critic_params,
            {
                "value_loss": value_loss,
                "actor_loss": actor_loss,
                "latent_prior_kl": latent_prior_kl,
                "critic_loss": critic_loss,
                "value": jnp.mean(value),
                "q_mean": jnp.mean(predicted_qs),
                "q_min": jnp.mean(jnp.min(predicted_qs, axis=0)),
                "advantage": jnp.mean(advantage),
                "adv_weight": jnp.mean(advantage_weight),
            },
        )

    return update

def make_iql_alignment_update(
    config: ConfigView,
    critic_optimizer: optax.GradientTransformation,
    q0_params: tuple[PyTree, ...],
):
    """Adapt ensemble Q geometry while preserving the selected IQL critics."""
    online_temperature = float(config.rlpd_initial_temperature)
    online_kl_weight = float(config.rlpd_latent_kl_weight)

    @jax.jit
    def update(
        critic_params,
        critic_opt_state,
        teacher_actor_params,
        obs,
        actor_sample_key,
    ):
        distribution = networks.gaussian_policy_fwd(teacher_actor_params, obs)
        latent_actions = distribution.sample(seed=actor_sample_key)
        teacher_score = -(
            latent_actions - distribution.loc
        ) / jnp.square(distribution.scale)
        score_target = jax.lax.stop_gradient(
            (online_temperature + online_kl_weight) * teacher_score
            + online_kl_weight * latent_actions
        )

        def q_mean_single(params, observation, latent):
            values = networks.q_ensemble_values(params, observation, latent)
            return jnp.mean(values, axis=0)

        def critic_loss_fn(params):
            predicted = networks.q_ensemble_values(params, obs, latent_actions)
            reference = networks.q_ensemble_values(q0_params, obs, latent_actions)
            current_q_grad = jax.vmap(
                jax.grad(q_mean_single, argnums=2), in_axes=(None, 0, 0)
            )(params, obs, latent_actions)
            score_loss = jnp.mean(jnp.square(current_q_grad - score_target))
            return score_loss, (predicted, reference, score_loss)

        (critic_loss, (predicted, reference, score_loss)), critic_grads = (
            jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_params)
        )
        critic_updates, critic_opt_state = critic_optimizer.update(
            critic_grads, critic_opt_state, critic_params
        )
        critic_params = optax.apply_updates(critic_params, critic_updates)
        after = networks.q_ensemble_values(critic_params, obs, latent_actions)
        reference_mean = jnp.mean(reference, axis=0)
        q_abs_drift = jnp.mean(jnp.abs(after - reference))
        q_relative_drift = q_abs_drift / (
            jnp.mean(jnp.abs(reference_mean)) + 1e-8
        )
        q_grad = jax.vmap(
            jax.grad(q_mean_single, argnums=2), in_axes=(None, 0, 0)
        )(critic_params, obs, latent_actions)
        q0_grad = jax.vmap(
            jax.grad(q_mean_single, argnums=2), in_axes=(None, 0, 0)
        )(q0_params, obs, latent_actions)
        q_grad_norm = jnp.mean(jnp.linalg.norm(q_grad, axis=-1))
        q_grad_cosine = jnp.mean(
            jnp.sum(q_grad * q0_grad, axis=-1)
            / (
                jnp.linalg.norm(q_grad, axis=-1)
                * jnp.linalg.norm(q0_grad, axis=-1)
                + 1e-8
            )
        )
        return critic_params, critic_opt_state, {
            "q_abs_drift": q_abs_drift,
            "q_relative_drift": q_relative_drift,
            "score_matching_loss": score_loss,
            "q_latent_grad_norm": q_grad_norm,
            "q_latent_grad_cosine_to_q0": q_grad_cosine,
        }

    return update

@jax.jit
def _iql_validation_kernel(
    expectile, discount, actor_nll_weight,
    actor_params, critic_params, value_params, target_critic_params,
    obs, next_obs, latents, rewards, masks,
):
    target_qs = networks.q_ensemble_values(target_critic_params, obs, latents)
    target_q = jnp.min(target_qs, axis=0)
    distribution = networks.gaussian_policy_fwd(actor_params, obs)
    actor_nll = -jnp.mean(jnp.sum(distribution.log_prob(latents), axis=-1))
    value = networks.value_mlp_fwd(value_params, obs)
    value_loss = jnp.mean(expectile_loss(target_q - value, expectile))
    next_value = networks.value_mlp_fwd(value_params, next_obs)
    bellman_target = rewards + discount * masks * next_value
    predicted_qs = networks.q_ensemble_values(critic_params, obs, latents)
    td_loss = jnp.mean(jnp.square(predicted_qs - bellman_target[None, :]))
    q_scale = jnp.maximum(jnp.mean(jnp.abs(bellman_target)), 1.0)
    value_scale = jnp.maximum(jnp.mean(jnp.abs(target_q)), 1.0)
    relative_td_rmse = jnp.sqrt(td_loss) / q_scale
    relative_value_rmse = jnp.sqrt(value_loss) / value_scale
    actor_nll_per_dim = actor_nll / latents.shape[-1]
    validation_score = (
        relative_td_rmse
        + relative_value_rmse
        + actor_nll_weight * actor_nll_per_dim
    )
    return jnp.asarray((
        td_loss, value_loss, td_loss + value_loss, q_scale, value_scale,
        relative_td_rmse, relative_value_rmse, actor_nll,
        actor_nll_per_dim, validation_score,
    ))


def iql_validation_losses(
    config: ConfigView,
    actor_params: PyTree,
    critic_params: tuple[PyTree, ...],
    value_params: PyTree,
    target_critic_params: tuple[PyTree, ...],
    normalized_observations: Array,
    normalized_next_observations: Array,
    rewards: Array,
    masks: Array,
    latent_targets: Array,
    indices: np.ndarray,
) -> dict[str, float]:
    """Compute held-out ensemble Bellman TD and expectile value losses."""
    values = jax.device_get(_iql_validation_kernel(
        config.expectile, config.discount,
        config.early_stopping_actor_nll_weight,
        actor_params, critic_params, value_params, target_critic_params,
        normalized_observations[jnp.asarray(indices)],
        normalized_next_observations[jnp.asarray(indices)],
        latent_targets[jnp.asarray(indices)],
        rewards[jnp.asarray(indices)],
        masks[jnp.asarray(indices)],
    ))
    names = (
        "validation_td_loss", "validation_value_loss", "validation_loss",
        "validation_q_scale", "validation_value_scale",
        "validation_relative_td_rmse", "validation_relative_value_rmse",
        "validation_actor_nll", "validation_actor_nll_per_dim",
        "validation_score",
    )
    return dict(zip(names, np.asarray(values).tolist()))

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


@jax.jit
def _policy_metrics_kernel(
    actor_params, critic_params, value_params, decoder,
    obs_norm, obs_raw, data_actions, targets,
):
    distribution = networks.gaussian_policy_fwd(actor_params, obs_norm)
    policy_z = distribution.loc
    policy_actions = forward_fm_batch(decoder, obs_raw, policy_z)
    q_policy_all = networks.q_ensemble_values(critic_params, obs_norm, policy_z)
    q_data_all = networks.q_ensemble_values(critic_params, obs_norm, targets)
    q_policy = jnp.min(q_policy_all, axis=0)
    q_data = jnp.min(q_data_all, axis=0)
    value = networks.value_mlp_fwd(value_params, obs_norm)
    return jnp.asarray((
        jnp.mean(q_policy), jnp.mean(q_data),
        jnp.mean(q_policy - q_data), jnp.mean(q_policy - value),
        jnp.mean(jnp.square(policy_actions - data_actions)),
        -jnp.mean(jnp.sum(distribution.log_prob(targets), axis=-1)),
        jnp.mean(jnp.square(policy_z - targets)),
        jnp.mean(distribution.scale),
        jnp.mean(jnp.std(q_policy_all, axis=0)),
    ))


def policy_metrics(
    actor_params: PyTree,
    critic_params: tuple[PyTree, ...],
    value_params: PyTree,
    decoder: Any,
    normalized_observations: Array,
    observations: Array,
    data_actions: Array,
    indices: np.ndarray,
    latent_targets: Array,
) -> dict[str, float]:
    values = jax.device_get(_policy_metrics_kernel(
        actor_params, critic_params, value_params, decoder,
        normalized_observations[jnp.asarray(indices)],
        observations[jnp.asarray(indices)],
        data_actions[jnp.asarray(indices)],
        latent_targets[jnp.asarray(indices)],
    ))
    names = (
        "comparison/policy_q", "comparison/data_q",
        "comparison/policy_q_minus_data_q", "comparison/policy_q_minus_v",
        "comparison/policy_action_data_mse", "encoder/latent_nll",
        "encoder/mean_target_mse", "encoder/scale_mean",
        "comparison/critic_std_policy",
    )
    return dict(zip(names, np.asarray(values).tolist()))

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
    critic_params: tuple[PyTree, ...],
    value_params: PyTree,
    decoder_epoch: int,
    encoder_iql_step: int,
    actor_opt_state: PyTree,
    critic_opt_state: PyTree,
    value_opt_state: PyTree,
    target_critic_params: tuple[PyTree, ...],
    rng: np.random.Generator,
    key: Array,
    bridge_metadata: dict[str, Any] | None = None,
) -> None:
    obs_dim = int(decoder.obs_stats.mean.shape[-1])
    action_dim = int(decoder.action_dim if hasattr(decoder, "action_dim") else decoder.params[-1][0].shape[-1])
    if rlpd_config.critic_ensemble_size != len(critic_params.backbones):
        raise ValueError(
            "Offline critic ensemble size does not match online RLPD config: "
            f"{len(critic_params.backbones)} != {rlpd_config.critic_ensemble_size}."
        )
    checkpoint = {
        "checkpoint_format": "gorl_offline_fm_rlpd",
        "checkpoint_version": 3,
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
        # Full RLPD-compatible state: all ensemble members are independently
        # initialized and trained offline before the online warm start.
        "rlpd_z_actor_params": actor_params,
        "rlpd_z_critic_params": critic_params,
        "rlpd_z_target_critic_params": target_critic_params,
        "rlpd_temperature_parameterization": "softplus_raw",
        "rlpd_z_log_temperature": jnp.log(
            jnp.expm1(jnp.asarray(rlpd_config.initial_temperature))
        ),
        "rlpd_z_obs_stats": encoder_obs_stats,
        # Offline-only training state/metadata.
        "decoder_type": config.decoder_type,
        "offline_config": dict(config),
        "iql_critic_params": critic_params,
        # Preserve twin-Q aliases for older analysis utilities.
        "q1_params": critic_params.backbones[0],
        "q2_params": critic_params.backbones[1],
        "value_params": value_params,
        "actor_opt_state": actor_opt_state,
        "critic_opt_state": critic_opt_state,
        "value_opt_state": value_opt_state,
        "target_iql_critic_params": target_critic_params,
        "target_q1_params": target_critic_params.backbones[0],
        "target_q2_params": target_critic_params.backbones[1],
        "numpy_rng_state": rng.bit_generator.state,
        "jax_key": key,
    }
    if bridge_metadata:
        checkpoint.update(bridge_metadata)
    with open(path, "wb") as file:
        pickle.dump(checkpoint, file)


def validate_online_rlpd_checkpoint(path: Path) -> None:
    """Verify an offline checkpoint can initialize the online RLPD pipeline."""
    with path.open("rb") as file:
        checkpoint = pickle.load(file)
    required = {
        "params", "obs_stats", "config", "obs_dim", "action_dim",
        "decoder_type", "rlpd_encoder_config",
        "rlpd_z_actor_params", "rlpd_z_critic_params",
        "rlpd_z_target_critic_params", "rlpd_z_log_temperature",
        "rlpd_z_obs_stats",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise RuntimeError(
            f"Saved offline checkpoint is not online-compatible; missing fields: {missing}"
        )



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
    env = None
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
                f"{environment_id} selected environment observation size "
                f"{env.observation_size}."
            )
        if env.action_size != action_dim:
            raise ValueError(
                f"Dataset action dim {action_dim} does not match "
                f"{environment_id} selected environment action size {env.action_size}."
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
        key, decoder_key, actor_key, value_key, critic_key = (
            jax.random.split(key, 5)
        )
        if config.decoder_type == "meanflow":
            decoder_config = Decoder1StepFMConfig(
                timestep_embed_dim=config.meanflow_timestep_embed_dim,
                hidden_dim=config.meanflow_hidden_dim,
                num_res_blocks=config.meanflow_num_res_blocks,
                mlp_expansion=config.meanflow_mlp_expansion,
                condition_type=config.meanflow_condition_type,
                policy_output_scale=config.meanflow_policy_output_scale,
                learning_rate=config.decoder_learning_rate,
                optimizer_beta1=config.meanflow_optimizer_beta1,
                optimizer_beta2=config.meanflow_optimizer_beta2,
                optimizer_eps=config.meanflow_optimizer_eps,
                optimizer_weight_decay=config.meanflow_optimizer_weight_decay,
                batch_size=config.decoder_batch_size,
                normalize_observations=config.meanflow_normalize_observations,
                normalization_mode=config.meanflow_normalization_mode,
                flow_ratio=config.meanflow_flow_ratio,
                time_dist=config.meanflow_time_dist,
                lognorm_mu=config.meanflow_lognorm_mu,
                lognorm_sigma=config.meanflow_lognorm_sigma,
                adaptive_loss_gamma=config.meanflow_adaptive_loss_gamma,
                adaptive_loss_c=config.meanflow_adaptive_loss_c,
                guidance_scale=config.meanflow_guidance_scale,
                use_dispersive=config.use_dispersive,
                dispersive_loss_weight=config.meanflow_dispersive_loss_weight,
                cycle_z_weight=config.meanflow_cycle_z_weight,
                cycle_a_weight=config.meanflow_cycle_a_weight,
                dispersive_tau=config.meanflow_dispersive_tau,
                dispersive_chunk_size=config.meanflow_dispersive_chunk_size,
                feather_std=config.meanflow_feather_std,
                latent_kl_weight=config.meanflow_latent_kl_weight,
            )
            decoder = Decoder1StepFMState.init(decoder_key, obs_dim, action_dim, decoder_config)
            with jdc.copy_and_mutate(decoder) as decoder:
                decoder.obs_stats = decoder.obs_stats.update(jnp.asarray(buffer.observations))
        else:
            decoder_config = DecoderFMConfig(
                flow_steps=config.flow_steps,
                timestep_embed_dim=config.timestep_embed_dim,
                hidden_dims=(config.decoder_hidden_size,) * config.decoder_num_layers,
                policy_output_scale=config.fm_policy_output_scale,
                learning_rate=config.decoder_learning_rate,
                batch_size=config.decoder_batch_size,
                num_epochs=config.decoder_max_epochs,
                n_samples_per_action=config.n_fm_samples_per_action,
                normalize_observations=config.fm_normalize_observations,
                sde_sigma=config.fm_sde_sigma,
                feather_std=config.fm_feather_std,
            )
            decoder = DecoderFMState.init(decoder_key, obs_dim, action_dim, decoder_config)
            with jdc.copy_and_mutate(decoder) as decoder:
                decoder.obs_stats = decoder.obs_stats.update(jnp.asarray(buffer.observations))
        if resume_checkpoint is not None:
            decoder = restore_decoder(decoder, resume_checkpoint)

        # IQL keeps its original actor/value training objectives, but the actor
        # layout matches EncoderState.init in encoder_rlpd.py so its parameters
        # can initialize online RLPD without conversion.
        actor_params = networks.gaussian_policy_init(
            actor_key,
            (obs_dim,)
            + (rlpd_config.hidden_size,) * rlpd_config.hidden_layers
            + (action_dim,),
        )
        value_params = networks.mlp_init(
            value_key, (obs_dim, 256, 256, 1), use_layer_norm=True
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
        # Keep the small offline dataset resident on the accelerator.  Batch
        # sampling below then avoids a NumPy copy and host-to-device transfer
        # on every update.
        device_observations = jax.device_put(buffer.observations)
        device_actions = jax.device_put(buffer.actions)
        device_rewards = jax.device_put(buffer.rewards)
        device_masks = jax.device_put(buffer.masks)
        device_normalized_observations = jax.device_put(normalized_observations)
        device_normalized_next_observations = jax.device_put(normalized_next_observations)

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
        critic_params = networks.critic_ensemble_init(
            critic_key, q_dims, rlpd_config.critic_ensemble_size
        )
        resume_encoder = (
            resume_checkpoint is not None
            and is_combined_checkpoint(resume_checkpoint)
        )
        start_iql_step = 0
        if resume_encoder:
            actor_params = resume_checkpoint["iql_z_actor_params"]
            value_params = resume_checkpoint["value_params"]
            if "iql_critic_params" not in resume_checkpoint:
                raise ValueError(
                    "The requested encoder checkpoint predates the independent "
                    "SERL-style critic ensemble and cannot resume this architecture."
                )
            critic_params = resume_checkpoint["iql_critic_params"]
            if rlpd_config.critic_ensemble_size != len(critic_params.backbones):
                raise ValueError("Resume checkpoint critic ensemble size mismatch.")
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
            device_normalized_observations = jax.device_put(normalized_observations)
            device_normalized_next_observations = jax.device_put(normalized_next_observations)
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
        _decoder_inverse, decoder_validation, decoder_metric_kernel = make_decoder_eval_functions(
            decoder,
            config.latent_inverse_steps if config.decoder_type == "flow_matching" else None,
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
                        epoch, device_observations[jnp.asarray(batch)], device_actions[jnp.asarray(batch)]
                    )
                else:
                    decoder, metrics = decoder.train_step(
                        device_observations[jnp.asarray(batch)],
                        # Train directly on dataset (s, a). Dataset actions are
                        # already the targets; never apply atanh/arctanh here.
                        device_actions[jnp.asarray(batch)],
                    )
                losses.append(metrics["loss"])
                global_step += 1
            validation_loss, key = decoder_validation_loss(
                decoder,
                device_observations,
                device_actions,
                validation_indices,
                config.decoder_eval_batches,
                key,
                decoder_validation,
            )
            comparison, current_latents = decoder_metrics(
                decoder,
                device_observations,
                device_actions,
                comparison_indices,
                decoder_metric_kernel,
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
                "decoder/train_cfm_loss": float(jax.device_get(jnp.mean(jnp.stack(losses)))),
                "decoder/validation_cfm_loss": validation_loss,
                **comparison,
            }
            append_metrics(metrics_path, record)
            logger.log(
                {k: v for k, v in record.items() if isinstance(v, (int, float))},
                global_step,
            )
            if validation_loss < best_validation:
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
            device_observations,
            device_actions,
            config.decoder_batch_size,
            config.latent_inverse_steps
            if config.decoder_type == "flow_matching"
            else None,
        )
        device_latent_targets = jax.device_put(latent_targets)
        frozen_metrics, _ = decoder_metrics(
            decoder,
            device_observations,
            device_actions,
            comparison_indices,
            decoder_metric_kernel,
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
        critic_opt_state = critic_optimizer.init(critic_params)
        value_opt_state = value_optimizer.init(value_params)
        target_critic_params = jax.tree.map(jnp.copy, critic_params)
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
            target_critic_params = resume_checkpoint.get(
                "target_iql_critic_params", target_critic_params
            )
        iql_update = make_iql_update(
            config, actor_optimizer, critic_optimizer, value_optimizer
        )
        accumulators: dict[str, Array] = {}
        accumulator_count = 0
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
                critic_params,
                critic_opt_state,
                value_params,
                value_opt_state,
                target_critic_params,
                metrics,
            ) = iql_update(
                actor_params,
                actor_opt_state,
                critic_params,
                critic_opt_state,
                value_params,
                value_opt_state,
                target_critic_params,
                device_normalized_observations[jnp.asarray(indices)],
                device_actions[jnp.asarray(indices)],
                device_rewards[jnp.asarray(indices)],
                device_normalized_next_observations[jnp.asarray(indices)],
                device_masks[jnp.asarray(indices)],
                device_latent_targets[jnp.asarray(indices)],
                (key := jax.random.fold_in(key, step)),
            )
            for name, value in metrics.items():
                accumulators[name] = accumulators.get(name, jnp.asarray(0.0)) + value
            accumulator_count += 1
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
                    critic_params,
                    value_params,
                    target_critic_params,
                    device_normalized_observations,
                    device_normalized_next_observations,
                    device_rewards,
                    device_masks,
                    device_latent_targets,
                    validation_eval_indices,
                )
                validation_score = latest_validation_metrics["validation_score"]
                if validation_score < best_validation_score:
                    # Always retain the best validation state. The minimum
                    # step only gates early stopping, not best-state tracking.
                    best_validation_score = validation_score
                    best_iql_step = completed_iql_steps
                    best_iql_state = (
                        actor_params, actor_opt_state,
                        critic_params, critic_opt_state,
                        value_params, value_opt_state,
                        target_critic_params,
                    )
                    stale_validations = 0
                elif completed_iql_steps >= config.early_stopping_min_steps:
                    stale_validations += 1
                # End of best-state update
                should_stop = (
                    completed_iql_steps >= config.early_stopping_min_steps
                    and stale_validations >= config.early_stopping_patience
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
                        f"iql/{name}": float(jax.device_get(values / accumulator_count))
                        for name, values in accumulators.items()
                    },
                    **policy_metrics(
                        actor_params,
                        critic_params,
                        value_params,
                        decoder,
                        device_normalized_observations,
                        device_observations,
                        device_actions,
                        policy_indices,
                        device_latent_targets,
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
                accumulator_count = 0

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
                    critic_params,
                    value_params,
                    decoder_epoch,
                    completed_iql_steps,
                    actor_opt_state,
                    critic_opt_state,
                    value_opt_state,
                    target_critic_params,
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
                critic_params, critic_opt_state,
                value_params, value_opt_state,
                target_critic_params,
            )
        (
            actor_params,
            actor_opt_state,
            critic_params, critic_opt_state,
            value_params, value_opt_state,
            target_critic_params,
        ) = best_iql_state
        print(
            f"Restored best encoder from IQL step {best_iql_step}."
        )
        # Persist the selected IQL optimum before alignment changes the critics.
        # This is deliberately separate from periodic and final checkpoints.
        iql_best_checkpoint_path = (
            output_dir / f"checkpoint_iql_best_step_{best_iql_step:09d}.pkl"
        )
        save_offline_checkpoint(
            iql_best_checkpoint_path,
            config,
            rlpd_config,
            decoder,
            actor_params,
            encoder_obs_stats,
            critic_params,
            value_params,
            decoder_epoch,
            best_iql_step,
            actor_opt_state,
            critic_opt_state,
            value_opt_state,
            target_critic_params,
            rng,
            key,
            bridge_metadata={
                "iql_best_checkpoint": True,
                "alignment_completed": False,
                "best_validation_score": best_validation_score,
            },
        )
        validate_online_rlpd_checkpoint(iql_best_checkpoint_path)
        print(
            "Saved and validated best IQL checkpoint before alignment: "
            f"{iql_best_checkpoint_path}"
        )

        # Alignment: freeze actor/value and distill both IQL critics while
        # adapting only their latent score geometry.
        teacher_actor_params = jax.tree.map(
            lambda value: jax.lax.stop_gradient(value), actor_params
        )
        q0_params = jax.tree.map(
            lambda value: jax.lax.stop_gradient(value), critic_params
        )
        alignment_steps = int(config.encoder_alignment_steps)
        if alignment_steps < 0:
            raise ValueError("encoder_alignment_steps must be non-negative.")
        if config.encoder_score_matching_weight < 0.0:
            raise ValueError("encoder_score_matching_weight must be non-negative.")
        alignment_accumulators: dict[str, Array] = {}
        alignment_total = 0
        alignment_count = 0
        alignment_base_step = global_step + completed_iql_steps
        alignment_update = make_iql_alignment_update(config, critic_optimizer, q0_params)
        for alignment_step in trange(
            alignment_steps, desc="IQL-to-RLPD bridge"
        ):
            indices = rng.choice(train_indices, size=config.batch_size, replace=True)
            (
                critic_params,
                critic_opt_state,
                alignment_metrics,
            ) = alignment_update(
                critic_params,
                critic_opt_state,
                teacher_actor_params,
                device_normalized_observations[jnp.asarray(indices)],
                (key := jax.random.fold_in(key, alignment_step)),
            )
            for name, value in alignment_metrics.items():
                alignment_accumulators[name] = (
                    alignment_accumulators.get(name, jnp.asarray(0.0)) + value
                )
            alignment_total += 1
            alignment_count += 1
            if (
                alignment_count % config.log_interval == 0
                or alignment_total == alignment_steps
            ):
                bridge_record = {
                    "method": "frozen_decoder",
                    "phase": "encoder_alignment",
                    "global_step": alignment_base_step + alignment_total,
                    "encoder/alignment_step": alignment_total,
                    **{
                        f"alignment/{name}": float(
                            jax.device_get(value / alignment_count)
                        )
                        for name, value in alignment_accumulators.items()
                    },
                }
                append_metrics(metrics_path, bridge_record)
                logger.log(
                    {
                        key: value
                        for key, value in bridge_record.items()
                        if isinstance(value, (int, float))
                    },
                    alignment_base_step + alignment_total,
                )
                print(f"  bridge step {alignment_total}: {bridge_record}")
                alignment_accumulators.clear()
                alignment_count = 0

        # Online must start with a target critic consistent with the aligned
        # critic; retaining the pre-alignment target would immediately pull Q
        # back toward Q0 during the first online updates.
        target_critic_params = critic_params

        decoder_path = output_dir / "checkpoint_final.pkl"
        final_iql_step = best_iql_step
        save_offline_checkpoint(
            decoder_path,
            config,
            rlpd_config,
            decoder,
            actor_params,
            encoder_obs_stats,
            critic_params,
            value_params,
            decoder_epoch,
            final_iql_step,
            actor_opt_state,
            critic_opt_state,
            value_opt_state,
            target_critic_params,
            rng,
            key,
            bridge_metadata={
                "encoder_alignment_steps": alignment_steps,
                "encoder_score_matching_weight": config.encoder_score_matching_weight,
                "encoder_alignment_q_preserve": True,
                "alignment_teacher_frozen": True,
            },
        )
        validate_online_rlpd_checkpoint(decoder_path)
        print(f"Saved and validated final offline RLPD checkpoint: {decoder_path}")
    finally:
        if env is not None:
            env.close()
        logger.finish()


if __name__ == "__main__":
    main(build_config(tyro.cli(TrainingConfig)))
