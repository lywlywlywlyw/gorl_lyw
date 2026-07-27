"""Online GoRL training on the Gymnasium environment matching a D4RL dataset.

This is a self-contained Gymnasium counterpart of ``scripts/run_gorl_fm.py``.
It directly resumes the encoder and frozen FM decoder written by
``run_offline_fm_frozen_new.py`` and repeats the same stage structure:

    PPO encoder update -> collect (state, decoded action, reward) ->
    FM decoder update -> next stage

Unlike the legacy pipeline, parameters are genuinely warm-started across the
offline-to-online boundary and across online stages.  The PPO optimizer is
initialized at the offline-to-online boundary because offline IQL and online
PPO use different optimizers and losses.
"""

from __future__ import annotations

import datetime
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import jax
import jax_dataclasses as jdc
import numpy as np
import optax
import tyro
from jax import Array
from jax import numpy as jnp
from tqdm import trange

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from flow_policy import encoder_ppo, networks
from flow_policy.decoder_fm import DecoderFMState


PyTree = Any


@dataclass
class OnlineConfig:
    """Configuration for D4RL-compatible online GoRL training."""

    offline_checkpoint: str
    output_dir: str = (
        "results/gorl_fm_d4rl_online_"
        + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    d4rl_dataset: str | None = None
    env_name: str | None = None
    seed: int = 1

    num_stages: int = 4
    encoder_num_timesteps: int = 100_000_000
    encoder_timesteps_per_stage: str | None = (
        "60000000,60000000,30000000,30000000"
    )

    # Gymnasium rollout and PPO.
    num_envs: int = 2048
    rollout_length: int = 480
    ppo_batch_size: int = 1024
    ppo_unroll_length: int = 30
    ppo_num_minibatches: int = 32
    ppo_epochs: int = 16
    encoder_learning_rate: float = 1e-3
    discounting: float = 0.995
    gae_lambda: float = 0.95
    entropy_cost: float = 0.01
    reward_scaling: float = 10.0
    value_loss_coeff: float = 0.25
    normalize_advantage: bool = True
    apply_tanh_in_rollout: bool = True
    episode_length: int = 1000
    num_evals: int = 10
    eval_episodes: int = 128

    # Stage-adaptive encoder settings, matching run_gorl_fm.py.
    z_regularization: float | None = None
    max_grad_norm: float = 0.5

    # Decoder collection and update.
    data_collection_iterations: int = 20
    collection_steps_per_iteration: int = 480
    fm_batch_size: int = 8192
    fm_num_epochs: int = 50
    fm_learning_rate: float = 3e-4
    fm_max_samples: int = 10_000_000
    fm_validation_fraction: float = 0.1
    fm_patience: int = 20
    fm_hidden_size: int = 64
    fm_num_layers: int = 4
    fm_hybrid_sampling: bool = False
    fm_high_quality_ratio: float = 0.8
    fm_high_quality_percentile: float = 0.5

    checkpoint_interval: int = 100_000


@dataclass
class RolloutBatch:
    observations: np.ndarray
    latents: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    values: np.ndarray
    next_observation: np.ndarray
    decoded_actions: np.ndarray


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a") as file:
        file.write(json.dumps(record) + "\n")


def d4rl_gymnasium_env_id(dataset_id: str) -> str:
    environment = dataset_id.split("-", 1)[0].lower()
    mapping = {
        "walker2d": "Walker2d-v5",
        "halfcheetah": "HalfCheetah-v5",
        "hopper": "Hopper-v5",
        "ant": "Ant-v5",
    }
    if environment not in mapping:
        raise ValueError(
            f"Unsupported D4RL MuJoCo dataset {dataset_id!r}; supported "
            f"families are {sorted(mapping)}."
        )
    return mapping[environment]


def parse_stage_timesteps(config: OnlineConfig) -> list[int]:
    if config.encoder_timesteps_per_stage is None:
        return [config.encoder_num_timesteps] * config.num_stages
    values = [
        int(value.strip())
        for value in config.encoder_timesteps_per_stage.split(",")
    ]
    if len(values) < config.num_stages:
        values.extend(
            [config.encoder_num_timesteps] * (config.num_stages - len(values))
        )
    return values[: config.num_stages]


def validate_config(config: OnlineConfig) -> None:
    positive = (
        config.num_stages,
        config.num_envs,
        config.rollout_length,
        config.ppo_batch_size,
        config.ppo_unroll_length,
        config.ppo_num_minibatches,
        config.ppo_epochs,
        config.num_evals,
        config.eval_episodes,
        config.episode_length,
        config.data_collection_iterations,
        config.collection_steps_per_iteration,
        config.fm_batch_size,
        config.fm_num_epochs,
        config.fm_patience,
        config.checkpoint_interval,
    )
    if min(positive) < 1:
        raise ValueError("Stage, rollout, batch, epoch and interval values must be positive.")
    if not 0.0 < config.fm_validation_fraction < 1.0:
        raise ValueError("fm_validation_fraction must be in (0, 1).")
    expected_rollout = (
        config.ppo_num_minibatches
        * config.ppo_batch_size
        * config.ppo_unroll_length
    )
    actual_rollout = config.num_envs * config.rollout_length
    if actual_rollout != expected_rollout:
        raise ValueError(
            "PPO rollout shape must match the original batching identity: "
            "num_envs * rollout_length == ppo_num_minibatches * "
            "ppo_batch_size * ppo_unroll_length. "
            f"Got {actual_rollout} != {expected_rollout}."
        )
    if not 0.0 <= config.fm_high_quality_ratio <= 1.0:
        raise ValueError("fm_high_quality_ratio must be in [0, 1].")
    if not 0.0 <= config.fm_high_quality_percentile <= 1.0:
        raise ValueError("fm_high_quality_percentile must be in [0, 1].")


def load_offline_checkpoint(
    config: OnlineConfig,
) -> tuple[
    dict[str, Any],
    encoder_ppo.ActorCriticParams,
    Any,
    DecoderFMState,
    str,
]:
    path = Path(config.offline_checkpoint).expanduser()
    with open(path, "rb") as file:
        checkpoint = pickle.load(file)
    required = {
        "params",
        "obs_stats",
        "config",
        "obs_dim",
        "action_dim",
        "ppo_z_params",
        "ppo_z_obs_stats",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(
            f"{path} is missing offline checkpoint fields: {sorted(missing)}. "
            "Pass checkpoint_final.pkl, not encoder_checkpoint_final.pkl."
        )

    offline_config = checkpoint.get("offline_config", {})
    dataset_id = config.d4rl_dataset or offline_config.get("d4rl_dataset")
    if config.env_name is not None:
        env_id = config.env_name
    elif dataset_id is not None:
        env_id = d4rl_gymnasium_env_id(dataset_id)
    else:
        checkpoint_env = checkpoint.get("env_name")
        if checkpoint_env is None:
            raise ValueError(
                "Could not determine an environment. Provide --d4rl-dataset "
                "or --env-name."
            )
        env_id = str(checkpoint_env)

    decoder = DecoderFMState.init(
        jax.random.key(config.seed + 1000),
        int(checkpoint["obs_dim"]),
        int(checkpoint["action_dim"]),
        checkpoint["config"],
    )
    with jdc.copy_and_mutate(decoder) as decoder:
        decoder.params = checkpoint["params"]
        decoder.obs_stats = checkpoint["obs_stats"]
        # Online decoder updates use the requested learning rate.
        decoder.config = jdc.replace(
            decoder.config, learning_rate=config.fm_learning_rate
        )
        decoder.opt = optax.adam(config.fm_learning_rate)
        decoder.opt_state = decoder.opt.init(decoder.params)
    expected_hidden_dims = (config.fm_hidden_size,) * config.fm_num_layers
    if tuple(decoder.config.hidden_dims) != expected_hidden_dims:
        raise ValueError(
            "Offline FM architecture does not match run_gorl_fm.py: "
            f"checkpoint hidden_dims={decoder.config.hidden_dims}, "
            f"expected={expected_hidden_dims}."
        )
    expected_fm = {
        "flow_steps": 10,
        "timestep_embed_dim": 8,
        "n_samples_per_action": 8,
        "batch_size": config.fm_batch_size,
        "policy_output_scale": 1.0,
        "normalize_observations": True,
        "sde_sigma": 0.0,
        "feather_std": 0.0,
    }
    for name, expected in expected_fm.items():
        actual = getattr(decoder.config, name)
        if actual != expected:
            raise ValueError(
                f"Offline FM config {name}={actual!r} does not match "
                f"run_gorl_fm.py value {expected!r}."
            )
    return (
        checkpoint,
        checkpoint["ppo_z_params"],
        checkpoint["ppo_z_obs_stats"],
        decoder,
        env_id,
    )


def make_vector_env(env_id: str, num_envs: int, seed: int):
    def make_one(index: int):
        def thunk():
            environment = gym.make(env_id)
            environment.reset(seed=seed + index)
            environment.action_space.seed(seed + index)
            return environment

        return thunk

    return gym.vector.SyncVectorEnv([make_one(i) for i in range(num_envs)])


def validate_shapes(
    env_id: str,
    observation_space: Any,
    action_space: Any,
    checkpoint: dict[str, Any],
    encoder_params: encoder_ppo.ActorCriticParams,
) -> None:
    obs_dim = int(checkpoint["obs_dim"])
    action_dim = int(checkpoint["action_dim"])
    if observation_space.shape != (obs_dim,):
        raise ValueError(
            f"{env_id} observation space {observation_space.shape} does not "
            f"match offline checkpoint obs_dim={obs_dim}."
        )
    if action_space.shape != (action_dim,):
        raise ValueError(
            f"{env_id} action space {action_space.shape} does not match "
            f"offline checkpoint action_dim={action_dim}."
        )
    policy_in = int(encoder_params.policy[0][0].shape[0])
    policy_out = int(encoder_params.policy[-1][0].shape[1])
    value_in = int(encoder_params.value[0][0].shape[0])
    if policy_in != obs_dim or value_in != obs_dim or policy_out != 2 * action_dim:
        raise ValueError(
            "Offline encoder parameter shapes do not match the environment: "
            f"policy input/output={policy_in}/{policy_out}, value input={value_in}, "
            f"expected={obs_dim}/{2 * action_dim}/{obs_dim}."
        )


@jax.jit
def policy_value(
    params: encoder_ppo.ActorCriticParams,
    obs_stats: Any,
    observations: Array,
) -> tuple[Any, Array]:
    normalized = (observations - obs_stats.mean) / (obs_stats.std + 1e-8)
    distribution = networks.gaussian_policy_fwd(params.policy, normalized)
    values = networks.value_mlp_fwd(params.value, normalized)
    return distribution, values


@jax.jit
def decode_actions(
    decoder: DecoderFMState, observations: Array, latents: Array
) -> Array:
    return decoder.sample_action_from_z(
        observations, latents, jax.random.key(0), deterministic=True
    )


def collect_rollout(
    env: Any,
    observations: np.ndarray,
    params: encoder_ppo.ActorCriticParams,
    obs_stats: Any,
    decoder: DecoderFMState,
    key: Array,
    length: int,
    apply_tanh: bool,
) -> tuple[RolloutBatch, np.ndarray, Array]:
    obs_rows = []
    latent_rows = []
    log_prob_rows = []
    reward_rows = []
    terminated_rows = []
    truncated_rows = []
    value_rows = []
    action_rows = []
    current_obs = observations

    for _ in range(length):
        key, sample_key = jax.random.split(key)
        distribution, values = policy_value(
            params, obs_stats, jnp.asarray(current_obs)
        )
        latents = distribution.sample(sample_key)
        log_probs = jnp.sum(distribution.log_prob(latents), axis=-1)
        decoded = decode_actions(
            decoder, jnp.asarray(current_obs), latents
        )
        decoded_np = np.asarray(decoded)
        env_actions = np.tanh(decoded_np) if apply_tanh else decoded_np
        env_actions = np.clip(
            env_actions, env.single_action_space.low, env.single_action_space.high
        )
        next_obs, rewards, terminated, truncated, _ = env.step(env_actions)

        obs_rows.append(np.asarray(current_obs, dtype=np.float32))
        latent_rows.append(np.asarray(latents, dtype=np.float32))
        log_prob_rows.append(np.asarray(log_probs, dtype=np.float32))
        reward_rows.append(np.asarray(rewards, dtype=np.float32))
        terminated_rows.append(np.asarray(terminated, dtype=np.float32))
        truncated_rows.append(np.asarray(truncated, dtype=np.float32))
        value_rows.append(np.asarray(values, dtype=np.float32))
        # Match collect_data_fm.py: save decoded action before environment tanh.
        action_rows.append(decoded_np.astype(np.float32))
        current_obs = np.asarray(next_obs, dtype=np.float32)

    return (
        RolloutBatch(
            observations=np.stack(obs_rows),
            latents=np.stack(latent_rows),
            log_probs=np.stack(log_prob_rows),
            rewards=np.stack(reward_rows),
            terminated=np.stack(terminated_rows),
            truncated=np.stack(truncated_rows),
            values=np.stack(value_rows),
            next_observation=current_obs,
            decoded_actions=np.stack(action_rows),
        ),
        current_obs,
        key,
    )


def compute_gae(
    rollout: RolloutBatch,
    last_values: np.ndarray,
    discounting: float,
    gae_lambda: float,
    reward_scaling: float,
) -> tuple[np.ndarray, np.ndarray]:
    truncation_mask = 1.0 - rollout.truncated
    # Match encoder_ppo.py: the transition discount first distinguishes true
    # termination, then the configured temporal discount is applied for GAE.
    # Time-limit truncations are handled separately by truncation_mask.
    discounts = (
        discounting * (1.0 - rollout.terminated)
    ).astype(np.float32)
    values_t_plus_1 = np.concatenate(
        [rollout.values[1:], last_values[None, :]], axis=0
    )
    scaled_rewards = rollout.rewards * reward_scaling
    deltas = (
        scaled_rewards + discounts * values_t_plus_1 - rollout.values
    ) * truncation_mask
    value_advantages = np.zeros_like(rollout.rewards, dtype=np.float32)
    accumulator = np.zeros_like(last_values, dtype=np.float32)
    for step in range(len(rollout.rewards) - 1, -1, -1):
        accumulator = deltas[step] + (
            discounts[step]
            * gae_lambda
            * truncation_mask[step]
            * accumulator
        )
        value_advantages[step] = accumulator
    value_targets = value_advantages + rollout.values
    targets_t_plus_1 = np.concatenate(
        [value_targets[1:], last_values[None, :]], axis=0
    )
    policy_advantages = (
        scaled_rewards
        + discounts * targets_t_plus_1
        - rollout.values
    ) * truncation_mask
    return policy_advantages, value_targets


def make_ppo_update(
    learning_rate: float,
    clipping_epsilon: float,
    entropy_cost: float,
    value_loss_coeff: float,
    z_regularization: float,
    max_grad_norm: float,
    optimizer: optax.GradientTransformation,
):
    @jax.jit
    def update(
        params,
        opt_state,
        obs_stats,
        observations,
        latents,
        old_log_probs,
        advantages,
        returns,
        value_masks,
    ):
        def loss_fn(current_params):
            normalized = (
                observations - obs_stats.mean
            ) / (obs_stats.std + 1e-8)
            distribution = networks.gaussian_policy_fwd(
                current_params.policy, normalized
            )
            new_log_probs = jnp.sum(
                distribution.log_prob(latents), axis=-1
            )
            ratios = jnp.exp(new_log_probs - old_log_probs)
            unclipped = ratios * advantages
            clipped = (
                jnp.clip(
                    ratios,
                    1.0 - clipping_epsilon,
                    1.0 + clipping_epsilon,
                )
                * advantages
            )
            policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
            values = networks.value_mlp_fwd(
                current_params.value, normalized
            )
            value_loss = value_loss_coeff * jnp.mean(
                jnp.square((returns - values) * value_masks)
            )
            entropy = jnp.mean(
                jnp.sum(distribution.entropy(), axis=-1)
            )
            z_reg_loss = z_regularization * (
                jnp.mean(jnp.square(distribution.loc))
                + jnp.mean(jnp.square(distribution.scale))
            )
            total = (
                policy_loss
                + value_loss
                - entropy_cost * entropy
                + z_reg_loss
            )
            return total, {
                "total_loss": total,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "z_reg_loss": z_reg_loss,
                "ratio": jnp.mean(ratios),
                "clip_fraction": jnp.mean(
                    jnp.abs(ratios - 1.0) > clipping_epsilon
                ),
            }

        (_, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        grad_norm = optax.global_norm(grads)
        if max_grad_norm > 0:
            scale = jnp.minimum(
                1.0, max_grad_norm / (grad_norm + 1e-8)
            )
            grads = jax.tree.map(lambda grad: grad * scale, grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        metrics["grad_norm"] = grad_norm
        metrics["learning_rate"] = jnp.asarray(learning_rate)
        return params, opt_state, metrics

    return update


def train_encoder_stage(
    *,
    config: OnlineConfig,
    stage: int,
    num_timesteps: int,
    env: Any,
    observations: np.ndarray,
    params: encoder_ppo.ActorCriticParams,
    obs_stats: Any,
    decoder: DecoderFMState,
    optimizer: optax.GradientTransformation,
    opt_state: PyTree,
    key: Array,
    metrics_path: Path,
    checkpoint_callback: Any,
    evaluation_callback: Any,
) -> tuple[Any, ...]:
    clipping = 0.15 if stage == 0 else 0.3
    max_grad_norm = config.max_grad_norm if stage == 0 else 1.0
    if config.z_regularization is not None:
        z_regularization = config.z_regularization
    else:
        z_regularization = 0.0005 if stage == 0 else 0.001
    update = make_ppo_update(
        config.encoder_learning_rate,
        clipping,
        config.entropy_cost,
        config.value_loss_coeff,
        z_regularization,
        max_grad_norm,
        optimizer,
    )
    interactions_per_rollout = config.num_envs * config.rollout_length
    num_rollouts = max(1, num_timesteps // interactions_per_rollout)
    completed = 0
    last_checkpoint = 0
    evaluation_rollouts = set(
        np.linspace(
            0, max(num_rollouts - 1, 0), config.num_evals, dtype=int
        ).tolist()
    )

    for rollout_index in trange(
        num_rollouts, desc=f"Stage {stage} PPO rollouts"
    ):
        if rollout_index in evaluation_rollouts:
            evaluation_callback(completed, params, obs_stats, decoder)
        rollout, observations, key = collect_rollout(
            env,
            observations,
            params,
            obs_stats,
            decoder,
            key,
            config.rollout_length,
            config.apply_tanh_in_rollout,
        )
        _, last_values = policy_value(
            params, obs_stats, jnp.asarray(rollout.next_observation)
        )
        advantages, returns = compute_gae(
            rollout,
            np.asarray(last_values),
            config.discounting,
            config.gae_lambda,
            config.reward_scaling,
        )
        flat_obs = rollout.observations.reshape(
            -1, rollout.observations.shape[-1]
        )
        flat_latents = rollout.latents.reshape(
            -1, rollout.latents.shape[-1]
        )
        flat_log_probs = rollout.log_probs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_value_masks = (1.0 - rollout.truncated).reshape(-1)
        if config.normalize_advantage:
            flat_advantages = (
                flat_advantages - flat_advantages.mean()
            ) / (flat_advantages.std() + 1e-8)

        obs_stats = obs_stats.update(jnp.asarray(flat_obs))
        metrics_accumulator: dict[str, list[float]] = {}
        sample_count = len(flat_obs)
        minibatch_size = config.ppo_batch_size * config.ppo_unroll_length
        update_rng = np.random.default_rng(
            config.seed + stage * 1_000_003 + rollout_index
        )
        for _ in range(config.ppo_epochs):
            permutation = update_rng.permutation(sample_count)
            for start in range(0, sample_count, minibatch_size):
                batch = permutation[start : start + minibatch_size]
                if len(batch) == 0:
                    continue
                params, opt_state, metrics = update(
                    params,
                    opt_state,
                    obs_stats,
                    jnp.asarray(flat_obs[batch]),
                    jnp.asarray(flat_latents[batch]),
                    jnp.asarray(flat_log_probs[batch]),
                    jnp.asarray(flat_advantages[batch]),
                    jnp.asarray(flat_returns[batch]),
                    jnp.asarray(flat_value_masks[batch]),
                )
                for name, value in metrics.items():
                    metrics_accumulator.setdefault(name, []).append(
                        float(value)
                    )

        completed += interactions_per_rollout
        record = {
            "phase": "encoder_ppo",
            "stage": stage,
            "stage_timesteps": min(completed, num_timesteps),
            "reward_mean": float(np.mean(rollout.rewards)),
            "z_mean": float(np.mean(rollout.latents)),
            "z_std": float(np.std(rollout.latents)),
            **{
                f"ppo/{name}": float(np.mean(values))
                for name, values in metrics_accumulator.items()
            },
        }
        append_jsonl(metrics_path, record)
        if completed - last_checkpoint >= config.checkpoint_interval:
            checkpoint_callback(
                min(completed, num_timesteps),
                params,
                obs_stats,
                decoder,
                opt_state,
            )
            last_checkpoint = completed
        if completed >= num_timesteps:
            break
    return observations, params, obs_stats, opt_state, key


def evaluate_policy(
    env_id: str,
    params: encoder_ppo.ActorCriticParams,
    obs_stats: Any,
    decoder: DecoderFMState,
    episodes: int,
    seed: int,
    apply_tanh: bool,
) -> dict[str, float]:
    returns = []
    lengths = []
    env = gym.make(env_id)
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            total = 0.0
            length = 0
            terminated = truncated = False
            while not (terminated or truncated):
                distribution, _ = policy_value(
                    params, obs_stats, jnp.asarray(obs[None, :])
                )
                latent = distribution.loc
                decoded = np.asarray(
                    decode_actions(
                        decoder, jnp.asarray(obs[None, :]), latent
                    )
                )[0]
                action = np.tanh(decoded) if apply_tanh else decoded
                action = np.clip(
                    action, env.action_space.low, env.action_space.high
                )
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                length += 1
            returns.append(total)
            lengths.append(length)
    finally:
        env.close()
    return {
        "reward_mean": float(np.mean(returns)),
        "reward_std": float(np.std(returns)),
        "reward_min": float(np.min(returns)),
        "reward_max": float(np.max(returns)),
        "episode_length_mean": float(np.mean(lengths)),
    }


def collect_decoder_data(
    config: OnlineConfig,
    env: Any,
    observations: np.ndarray,
    params: encoder_ppo.ActorCriticParams,
    obs_stats: Any,
    decoder: DecoderFMState,
    key: Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Array]:
    states = []
    actions = []
    rewards = []
    for _ in trange(
        config.data_collection_iterations, desc="Collect decoder data"
    ):
        rollout, observations, key = collect_rollout(
            env,
            observations,
            params,
            obs_stats,
            decoder,
            key,
            config.collection_steps_per_iteration,
            config.apply_tanh_in_rollout,
        )
        states.append(
            rollout.observations.reshape(-1, rollout.observations.shape[-1])
        )
        actions.append(
            rollout.decoded_actions.reshape(
                -1, rollout.decoded_actions.shape[-1]
            )
        )
        rewards.append(rollout.rewards.reshape(-1))
    return (
        np.concatenate(states),
        np.concatenate(actions),
        np.concatenate(rewards),
        observations,
        key,
    )


def decoder_validation_loss(
    decoder: DecoderFMState,
    observations: np.ndarray,
    actions: np.ndarray,
    key: Array,
) -> tuple[float, Array]:
    losses = []
    action_dim = actions.shape[-1]
    number_of_batches = min(
        50, len(observations) // decoder.config.batch_size
    )
    if number_of_batches == 0:
        number_of_batches = 1
    for batch_index in range(number_of_batches):
        start = batch_index * decoder.config.batch_size
        obs = jnp.asarray(
            observations[start : start + decoder.config.batch_size]
        )
        act = jnp.asarray(actions[start : start + decoder.config.batch_size])
        if len(obs) == 0:
            continue
        normalized = (
            obs - decoder.obs_stats.mean
        ) / (decoder.obs_stats.std + 1e-8)
        key, eps_key, time_key = jax.random.split(key, 3)
        eps = jax.random.normal(
            eps_key,
            (len(obs), decoder.config.n_samples_per_action, action_dim),
        )
        times = jax.random.uniform(
            time_key,
            (len(obs), decoder.config.n_samples_per_action, 1),
        )
        losses.append(
            float(
                jnp.mean(
                    decoder.compute_cfm_loss(
                        normalized, act, eps, times
                    )
                )
            )
        )
    return float(np.mean(losses)), key


def train_decoder_stage(
    config: OnlineConfig,
    decoder: DecoderFMState,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    key: Array,
    stage: int,
    metrics_path: Path,
) -> tuple[DecoderFMState, Array]:
    rng = np.random.default_rng(config.seed + stage + 10_000)
    if config.fm_hybrid_sampling:
        number_of_episodes = len(observations) // config.episode_length
        if number_of_episodes < 1:
            raise ValueError(
                "Hybrid sampling needs at least one complete episode."
            )
        usable = number_of_episodes * config.episode_length
        observations = observations[:usable]
        actions = actions[:usable]
        rewards = rewards[:usable]
        episode_rewards = rewards.reshape(
            number_of_episodes, config.episode_length
        ).sum(axis=1)
        threshold = np.percentile(
            episode_rewards, config.fm_high_quality_percentile * 100
        )
        high_quality = np.flatnonzero(episode_rewards >= threshold)
        high_quality_count = int(
            number_of_episodes * config.fm_high_quality_ratio
        )
        coverage_count = number_of_episodes - high_quality_count
        selected_high_quality = rng.choice(
            high_quality,
            high_quality_count,
            replace=len(high_quality) < high_quality_count,
        )
        selected_coverage = rng.choice(
            number_of_episodes, coverage_count, replace=False
        )
        selected_episodes = np.concatenate(
            [selected_high_quality, selected_coverage]
        )
        selected_rows = (
            selected_episodes[:, None] * config.episode_length
            + np.arange(config.episode_length)[None, :]
        ).reshape(-1)
        observations = observations[selected_rows]
        actions = actions[selected_rows]
        rewards = rewards[selected_rows]
    if len(observations) > config.fm_max_samples:
        selected = rng.choice(
            len(observations), config.fm_max_samples, replace=False
        )
        observations = observations[selected]
        actions = actions[selected]
        rewards = rewards[selected]
    permutation = rng.permutation(len(observations))
    validation_size = max(
        1, int(len(observations) * config.fm_validation_fraction)
    )
    validation_indices = permutation[:validation_size]
    training_indices = permutation[validation_size:]
    if len(training_indices) == 0:
        raise ValueError("Not enough collected samples for decoder training.")

    # Continue from the current decoder, while incorporating online states in
    # its observation normalization statistics.
    with jdc.copy_and_mutate(decoder) as decoder:
        decoder.obs_stats = decoder.obs_stats.update(
            jnp.asarray(observations[training_indices])
        )
    best_params = jax.tree.map(jnp.copy, decoder.params)
    best_validation = float("inf")
    stale = 0
    for epoch in trange(
        config.fm_num_epochs, desc=f"Stage {stage} FM epochs"
    ):
        losses = []
        shuffled = rng.permutation(training_indices)
        number_of_batches = len(shuffled) // config.fm_batch_size
        if number_of_batches == 0:
            raise ValueError(
                "Collected decoder training data is smaller than "
                f"fm_batch_size={config.fm_batch_size}."
            )
        for batch_index in range(number_of_batches):
            start = batch_index * config.fm_batch_size
            batch = shuffled[start : start + config.fm_batch_size]
            decoder, metrics = decoder.train_step(
                jnp.asarray(observations[batch]),
                jnp.asarray(actions[batch]),
            )
            losses.append(float(metrics["loss"]))
        validation, key = decoder_validation_loss(
            decoder,
            observations[validation_indices],
            actions[validation_indices],
            key,
        )
        append_jsonl(
            metrics_path,
            {
                "phase": "decoder_fm",
                "stage": stage,
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation,
            },
        )
        if validation < best_validation:
            best_validation = validation
            best_params = jax.tree.map(jnp.copy, decoder.params)
            stale = 0
        else:
            stale += 1
            if stale >= config.fm_patience:
                break
    with jdc.copy_and_mutate(decoder) as decoder:
        decoder.params = best_params
    return decoder, key


def save_checkpoint(
    path: Path,
    config: OnlineConfig,
    env_id: str,
    stage: int,
    stage_timesteps: int,
    params: encoder_ppo.ActorCriticParams,
    obs_stats: Any,
    decoder: DecoderFMState,
    encoder_opt_state: PyTree,
    best_reward: float,
) -> None:
    checkpoint = {
        # Same combined keys written by the offline script.
        "params": decoder.params,
        "obs_stats": decoder.obs_stats,
        "config": decoder.config,
        "obs_dim": int(decoder.obs_stats.mean.shape[-1]),
        "action_dim": int(decoder.params[-1][0].shape[-1]),
        "ppo_z_params": params,
        "ppo_z_obs_stats": obs_stats,
        "fm_params": decoder.params,
        "fm_obs_stats": decoder.obs_stats,
        "env_name": env_id,
        "d4rl_dataset": config.d4rl_dataset,
        "decoder_type": "fm",
        "z_dim": int(decoder.params[-1][0].shape[-1]),
        # Resume metadata.
        "online_config": asdict(config),
        "online_stage": stage,
        "stage_timesteps": stage_timesteps,
        "encoder_opt_state": encoder_opt_state,
        "best_reward": best_reward,
    }
    with open(path, "wb") as file:
        pickle.dump(checkpoint, file)


def main(config: OnlineConfig) -> None:
    validate_config(config)
    stage_timesteps = parse_stage_timesteps(config)
    (
        offline_checkpoint,
        encoder_params,
        encoder_obs_stats,
        decoder,
        env_id,
    ) = load_offline_checkpoint(config)
    if config.d4rl_dataset is None:
        config.d4rl_dataset = offline_checkpoint.get(
            "offline_config", {}
        ).get("d4rl_dataset")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    with open(output_dir / "config.json", "w") as file:
        json.dump(
            {
                **asdict(config),
                "resolved_environment": env_id,
                "method": "GoRL FM staged online training",
            },
            file,
            indent=2,
        )

    env = make_vector_env(
        env_id, config.num_envs, config.seed
    )
    try:
        validate_shapes(
            env_id,
            env.single_observation_space,
            env.single_action_space,
            offline_checkpoint,
            encoder_params,
        )
        observations, _ = env.reset(seed=config.seed)
        observations = np.asarray(observations, dtype=np.float32)
        key = jax.random.key(config.seed)
        encoder_optimizer = optax.adam(config.encoder_learning_rate)
        encoder_opt_state = encoder_optimizer.init(encoder_params)
        best_reward = -float("inf")

        print(
            f"Resuming offline checkpoint {config.offline_checkpoint}\n"
            f"Environment: {env_id}, obs_dim={observations.shape[-1]}, "
            f"action_dim={env.single_action_space.shape[-1]}\n"
            f"Stages: {config.num_stages}, timesteps={stage_timesteps}"
        )

        for stage in range(config.num_stages):
            stage_dir = output_dir / f"stage_{stage}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            stage_best_reward = -float("inf")
            stage_best_params = encoder_params
            stage_best_obs_stats = encoder_obs_stats

            def periodic_save(
                completed,
                current_params,
                current_stats,
                current_decoder,
                current_opt_state,
            ):
                save_checkpoint(
                    stage_dir
                    / f"checkpoint_step_{completed:012d}.pkl",
                    config,
                    env_id,
                    stage,
                    completed,
                    current_params,
                    current_stats,
                    current_decoder,
                    current_opt_state,
                    best_reward,
                )

            def stage_evaluate(
                completed,
                current_params,
                current_stats,
                current_decoder,
            ):
                nonlocal best_reward, stage_best_reward
                nonlocal stage_best_params, stage_best_obs_stats
                evaluation = evaluate_policy(
                    env_id,
                    current_params,
                    current_stats,
                    current_decoder,
                    config.eval_episodes,
                    config.seed + stage * 100_000 + completed,
                    config.apply_tanh_in_rollout,
                )
                best_reward = max(
                    best_reward, evaluation["reward_mean"]
                )
                if evaluation["reward_mean"] >= stage_best_reward - 1e-6:
                    stage_best_reward = evaluation["reward_mean"]
                    stage_best_params = jax.tree.map(
                        jnp.copy, current_params
                    )
                    stage_best_obs_stats = jax.tree.map(
                        jnp.copy, current_stats
                    )
                append_jsonl(
                    metrics_path,
                    {
                        "phase": "evaluation",
                        "stage": stage,
                        "stage_timesteps": completed,
                        **evaluation,
                    },
                )
                print(
                    f"Stage {stage} evaluation at {completed}: {evaluation}"
                )

            (
                observations,
                encoder_params,
                encoder_obs_stats,
                encoder_opt_state,
                key,
            ) = train_encoder_stage(
                config=config,
                stage=stage,
                num_timesteps=stage_timesteps[stage],
                env=env,
                observations=observations,
                params=encoder_params,
                obs_stats=encoder_obs_stats,
                decoder=decoder,
                optimizer=encoder_optimizer,
                opt_state=encoder_opt_state,
                key=key,
                metrics_path=metrics_path,
                checkpoint_callback=periodic_save,
                evaluation_callback=stage_evaluate,
            )
            # The original pipeline collects data from best_checkpoint.pkl,
            # rather than from the final PPO iteration.
            encoder_params = stage_best_params
            encoder_obs_stats = stage_best_obs_stats
            encoder_opt_state = encoder_optimizer.init(encoder_params)

            (
                collected_obs,
                collected_actions,
                collected_rewards,
                observations,
                key,
            ) = collect_decoder_data(
                config,
                env,
                observations,
                encoder_params,
                encoder_obs_stats,
                decoder,
                key,
            )
            with open(stage_dir / "online_decoder_data.pkl", "wb") as file:
                pickle.dump(
                    {
                        "states": collected_obs,
                        "actions": collected_actions,
                        "rewards": collected_rewards,
                        "env_name": env_id,
                        "d4rl_dataset": config.d4rl_dataset,
                    },
                    file,
                )

            decoder, key = train_decoder_stage(
                config,
                decoder,
                collected_obs,
                collected_actions,
                collected_rewards,
                key,
                stage,
                metrics_path,
            )
            stage_checkpoint = stage_dir / "checkpoint_final.pkl"
            save_checkpoint(
                stage_checkpoint,
                config,
                env_id,
                stage,
                stage_timesteps[stage],
                encoder_params,
                encoder_obs_stats,
                decoder,
                encoder_opt_state,
                best_reward,
            )
            print(f"Saved stage {stage} checkpoint: {stage_checkpoint}")

        final_path = output_dir / "checkpoint_final.pkl"
        save_checkpoint(
            final_path,
            config,
            env_id,
            config.num_stages - 1,
            stage_timesteps[-1],
            encoder_params,
            encoder_obs_stats,
            decoder,
            encoder_opt_state,
            best_reward,
        )
        print(f"Online GoRL training complete: {final_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main(tyro.cli(OnlineConfig))
