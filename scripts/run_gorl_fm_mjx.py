"""GoRL FM online training on D4RL-family Brax environments with MJX.

This entrypoint uses the PPO, agent, rollout, network, and FM implementations
from ``src/flow_policy`` that are also used by ``scripts/run_gorl_fm.py``.
It adds only the missing D4RL-task-to-MJX adapter and offline-checkpoint
warm-start/stage orchestration.
"""

from __future__ import annotations

import datetime
import json
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax_dataclasses as jdc
import numpy as np
import tyro
from brax import envs as brax_envs
from jax import numpy as jnp
from tqdm import trange

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flow_policy import encoder_ppo
from flow_policy.agent import EncoderFMAgent
from flow_policy.decoder_fm import DecoderFMConfig, DecoderFMState
from flow_policy import rollouts


@dataclass
class Config:
    offline_checkpoint: str | None = None
    output_dir: str = (
        "results/gorl_fm_mjx_online_"
        + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    d4rl_dataset: str | None = "walker2d-medium-expert-v2"
    env_name: str | None = None
    seed: int = 1
    num_stages: int = 4
    encoder_num_timesteps: int = 100_000_000
    encoder_timesteps_per_stage: str | None = (
        "60000000,60000000,30000000,30000000"
    )

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

    z_regularization: float | None = None
    max_grad_norm: float = 0.5

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

    wandb_enabled: bool = True
    wandb_project: str = "GoRL-online"
    wandb_entity: str | None = None
    wandb_run_name: str | None = (
        "gorl_onine_"
        + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    wandb_mode: str = "online"
    wandb_group: str | None = None
    wandb_tags: tuple[str, ...] = ()


@jdc.pytree_dataclass
class BraxRolloutState:
    """Brax-State equivalent of flow_policy's MJX Playground rollout state."""

    env: jdc.Static[Any]
    env_state: Any
    first_obs: jax.Array
    first_pipeline_state: Any
    steps: jax.Array
    num_envs: jdc.Static[int]
    prng: jax.Array

    @staticmethod
    @jdc.jit
    def init(
        env: jdc.Static[Any], prng: jax.Array, num_envs: jdc.Static[int]
    ) -> "BraxRolloutState":
        prng, reset_prng = jax.random.split(prng)
        state = jax.vmap(env.reset)(
            jax.random.split(reset_prng, num=num_envs)
        )
        return BraxRolloutState(
            env=env,
            env_state=state,
            first_obs=state.obs,
            first_pipeline_state=state.pipeline_state,
            steps=jnp.zeros_like(state.done),
            num_envs=num_envs,
            prng=prng,
        )

    @jdc.jit
    def rollout(
        self,
        agent: EncoderFMAgent,
        episode_length: jdc.Static[int],
        iterations_per_env: jdc.Static[int],
        auto_reset: jdc.Static[bool] = True,
        deterministic: jdc.Static[bool] = False,
        apply_tanh_in_rollout: jdc.Static[bool] = True,
    ) -> tuple["BraxRolloutState", Any]:
        def step(carry: BraxRolloutState, _):
            key, next_key = jax.random.split(carry.prng)
            z, z_info = agent.sample_z(
                carry.env_state.obs, key, deterministic=deterministic
            )
            action = agent.map_z_to_action(carry.env_state.obs, z)
            env_action = (
                jnp.tanh(action) if apply_tanh_in_rollout else action
            )
            next_state = jax.vmap(carry.env.step)(
                carry.env_state, env_action
            )
            next_steps = carry.steps + 1
            truncation = next_steps >= episode_length
            done = next_state.done.astype(bool)
            done_or_truncation = jnp.logical_or(done, truncation)
            transition = rollouts.TransitionStruct(
                obs=carry.env_state.obs,
                next_obs=next_state.obs,
                action=z,
                action_info=z_info,
                reward=next_state.reward,
                truncation=truncation.astype(jnp.float32),
                discount=1.0 - done.astype(jnp.float32),
            )
            if auto_reset:
                def choose(first, current):
                    mask = done_or_truncation.reshape(
                        done_or_truncation.shape
                        + (1,) * (current.ndim - done_or_truncation.ndim)
                    )
                    return jnp.where(mask, first, current)

                next_state = next_state.replace(
                    obs=jax.tree.map(
                        choose, carry.first_obs, next_state.obs
                    ),
                    pipeline_state=jax.tree.map(
                        choose,
                        carry.first_pipeline_state,
                        next_state.pipeline_state,
                    ),
                    done=jnp.zeros_like(next_state.done),
                )
                next_steps = jnp.where(
                    done_or_truncation, 0, next_steps
                )
            with jdc.copy_and_mutate(carry) as updated:
                updated.env_state = next_state
                updated.steps = next_steps
                updated.prng = next_key
            return updated, transition

        return jax.lax.scan(
            step, self, xs=None, length=iterations_per_env
        )

    @jdc.jit
    def rollout_with_actions(
        self,
        agent: EncoderFMAgent,
        episode_length: jdc.Static[int],
        iterations_per_env: jdc.Static[int],
        apply_tanh_in_rollout: jdc.Static[bool] = True,
    ) -> tuple["BraxRolloutState", jax.Array, jax.Array, jax.Array]:
        def step(carry: BraxRolloutState, _):
            key, next_key = jax.random.split(carry.prng)
            z, _ = agent.sample_z(
                carry.env_state.obs, key, deterministic=False
            )
            action = agent.map_z_to_action(carry.env_state.obs, z)
            env_action = (
                jnp.tanh(action) if apply_tanh_in_rollout else action
            )
            next_state = jax.vmap(carry.env.step)(
                carry.env_state, env_action
            )
            next_steps = carry.steps + 1
            done = jnp.logical_or(
                next_state.done.astype(bool),
                next_steps >= episode_length,
            )

            def choose(first, current):
                mask = done.reshape(
                    done.shape + (1,) * (current.ndim - done.ndim)
                )
                return jnp.where(mask, first, current)

            reset_state = next_state.replace(
                obs=jax.tree.map(choose, carry.first_obs, next_state.obs),
                pipeline_state=jax.tree.map(
                    choose,
                    carry.first_pipeline_state,
                    next_state.pipeline_state,
                ),
                done=jnp.zeros_like(next_state.done),
            )
            with jdc.copy_and_mutate(carry) as updated:
                updated.env_state = reset_state
                updated.steps = jnp.where(done, 0, next_steps)
                updated.prng = next_key
            return updated, (carry.env_state.obs, action, next_state.reward)

        state, (obs, actions, rewards) = jax.lax.scan(
            step, self, xs=None, length=iterations_per_env
        )
        return state, obs, actions, rewards


def parse_timesteps(config: Config) -> list[int]:
    if config.encoder_timesteps_per_stage is None:
        return [config.encoder_num_timesteps] * config.num_stages
    values = [
        int(x.strip()) for x in config.encoder_timesteps_per_stage.split(",")
    ]
    values.extend(
        [config.encoder_num_timesteps]
        * max(0, config.num_stages - len(values))
    )
    return values[: config.num_stages]


def validate_config(config: Config) -> None:
    values = (
        config.num_stages,
        config.num_envs,
        config.rollout_length,
        config.ppo_batch_size,
        config.ppo_unroll_length,
        config.ppo_num_minibatches,
        config.ppo_epochs,
        config.eval_episodes,
        config.fm_batch_size,
        config.fm_num_epochs,
        config.checkpoint_interval,
    )
    if min(values) < 1:
        raise ValueError("Training counts and sizes must be positive.")
    lhs = config.num_envs * config.rollout_length
    rhs = (
        config.ppo_num_minibatches
        * config.ppo_batch_size
        * config.ppo_unroll_length
    )
    if lhs != rhs:
        raise ValueError(
            "num_envs * rollout_length must equal ppo_num_minibatches * "
            f"ppo_batch_size * ppo_unroll_length; got {lhs} != {rhs}."
        )
    if not 0 < config.fm_validation_fraction < 1:
        raise ValueError("fm_validation_fraction must be in (0, 1).")


def resolve_task(checkpoint: dict[str, Any], config: Config) -> tuple[str, str]:
    offline_config = checkpoint.get("offline_config", {})
    dataset = (
        config.d4rl_dataset
        or checkpoint.get("d4rl_dataset")
        or offline_config.get("d4rl_dataset")
    )
    source = config.env_name or dataset or checkpoint.get("env_name")
    if source is None:
        raise ValueError("Provide --env-name or --d4rl-dataset.")
    family = str(source).split("-", 1)[0].lower()
    mapping = {
        "walker2d": "walker2d",
        "hopper": "hopper",
        "halfcheetah": "halfcheetah",
        "ant": "ant",
    }
    if family not in mapping:
        raise ValueError(
            f"Unsupported D4RL MJX task {source!r}; supported: {sorted(mapping)}"
        )
    return str(source), mapping[family]


def encoder_config(
    config: Config,
    num_timesteps: int,
    action_dim: int,
    stage: int,
) -> encoder_ppo.EncoderConfig:
    z_reg = config.z_regularization
    if z_reg is None:
        z_reg = 0.0005 if stage == 0 else 0.001
    return encoder_ppo.EncoderConfig(
        action_repeat=1,
        batch_size=config.ppo_batch_size,
        discounting=config.discounting,
        entropy_cost=config.entropy_cost,
        episode_length=config.episode_length,
        learning_rate=config.encoder_learning_rate,
        normalize_observations=True,
        num_envs=config.num_envs,
        num_evals=config.num_evals,
        num_minibatches=config.ppo_num_minibatches,
        num_timesteps=num_timesteps,
        num_updates_per_batch=config.ppo_epochs,
        reward_scaling=config.reward_scaling,
        unroll_length=config.ppo_unroll_length,
        z_dim=action_dim,
        clipping_epsilon=0.15 if stage == 0 else 0.3,
        gae_lambda=config.gae_lambda,
        normalize_advantage=config.normalize_advantage,
        value_loss_coeff=config.value_loss_coeff,
        z_regularization=z_reg,
        max_grad_norm=config.max_grad_norm if stage == 0 else 1.0,
    )


def load_states(
    path: str | None,
    env: Any,
    config: Config,
    stage_timesteps: int,
) -> tuple[dict[str, Any], encoder_ppo.EncoderState, DecoderFMState]:
    action_dim = int(env.action_size)
    obs_dim = int(env.observation_size)
    ppo_config = encoder_config(config, stage_timesteps, action_dim, 0)

    if path is None:
        encoder = encoder_ppo.EncoderState.init(
            jax.random.key(config.seed), env, ppo_config
        )
        decoder_config = DecoderFMConfig(
            flow_steps=10,
            timestep_embed_dim=8,
            hidden_dims=(config.fm_hidden_size,) * config.fm_num_layers,
            policy_output_scale=1.0,
            learning_rate=config.fm_learning_rate,
            batch_size=config.fm_batch_size,
            num_epochs=config.fm_num_epochs,
            n_samples_per_action=8,
            normalize_observations=True,
            sde_sigma=0.0,
            feather_std=0.0,
        )
        decoder = DecoderFMState.init(
            jax.random.PRNGKey(config.seed + 1000),
            obs_dim,
            action_dim,
            decoder_config,
        )
        return {}, encoder, decoder

    with open(Path(path).expanduser(), "rb") as file:
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
        raise KeyError(f"Offline checkpoint missing keys: {sorted(missing)}")
    checkpoint_obs_dim = int(checkpoint["obs_dim"])
    checkpoint_action_dim = int(checkpoint["action_dim"])
    if (
        env.observation_size != checkpoint_obs_dim
        or env.action_size != checkpoint_action_dim
    ):
        raise ValueError(
            "MJX/checkpoint shape mismatch: "
            f"env={env.observation_size}/{env.action_size}, "
            f"checkpoint={checkpoint_obs_dim}/{checkpoint_action_dim}."
        )
    encoder = encoder_ppo.EncoderState.init(
        jax.random.key(config.seed), env, ppo_config
    )
    with jdc.copy_and_mutate(encoder) as encoder:
        encoder.params = checkpoint["ppo_z_params"]
        encoder.obs_stats = checkpoint["ppo_z_obs_stats"]

    decoder = DecoderFMState.init(
        jax.random.PRNGKey(config.seed + 1000),
        checkpoint_obs_dim,
        checkpoint_action_dim,
        checkpoint["config"],
    )
    with jdc.copy_and_mutate(decoder) as decoder:
        decoder.params = checkpoint["params"]
        decoder.obs_stats = checkpoint["obs_stats"]
        decoder.config = jdc.replace(
            decoder.config,
            learning_rate=config.fm_learning_rate,
            batch_size=config.fm_batch_size,
        )
        decoder.opt = __import__("optax").adam(config.fm_learning_rate)
        decoder.opt_state = decoder.opt.init(decoder.params)
    expected_hidden = (config.fm_hidden_size,) * config.fm_num_layers
    if tuple(decoder.config.hidden_dims) != expected_hidden:
        raise ValueError(
            f"FM hidden_dims={decoder.config.hidden_dims}, "
            f"expected={expected_hidden}."
        )
    return checkpoint, encoder, decoder


def append_jsonl(path: Path, record: dict[str, Any], run: Any) -> None:
    with open(path, "a") as file:
        file.write(json.dumps(record) + "\n")
    if run is not None:
        run.log(record)


def evaluate(
    agent: EncoderFMAgent,
    config: Config,
    key: jax.Array,
) -> dict[str, float]:
    rollout = BraxRolloutState.init(
        agent.ppo_z_state.env,
        key,
        config.eval_episodes,
    )
    _, transitions = rollout.rollout(
        agent,
        episode_length=config.episode_length,
        iterations_per_env=config.episode_length,
        auto_reset=False,
        deterministic=True,
        apply_tanh_in_rollout=config.apply_tanh_in_rollout,
    )
    ended = jnp.logical_or(
        transitions.discount == 0, transitions.truncation > 0
    )
    active = jnp.concatenate(
        [
            jnp.ones_like(ended[:1], dtype=bool),
            jnp.cumprod(~ended[:-1], axis=0).astype(bool),
        ],
        axis=0,
    )
    returns = jnp.sum(transitions.reward * active, axis=0)
    lengths = jnp.sum(active, axis=0)
    return {
        "reward_mean": float(jnp.mean(returns)),
        "reward_std": float(jnp.std(returns)),
        "reward_min": float(jnp.min(returns)),
        "reward_max": float(jnp.max(returns)),
        "steps_mean": float(jnp.mean(lengths)),
        "steps_std": float(jnp.std(lengths)),
        "steps_min": float(jnp.min(lengths)),
        "steps_max": float(jnp.max(lengths)),
    }


def save_checkpoint(
    path: Path,
    checkpoint: dict[str, Any],
    config: Config,
    source_env: str,
    stage: int,
    steps: int,
    agent: EncoderFMAgent,
    best_reward: float,
) -> None:
    decoder = agent.fm_state
    payload = {
        "params": decoder.params,
        "obs_stats": decoder.obs_stats,
        "config": decoder.config,
        "obs_dim": int(decoder.obs_stats.mean.shape[-1]),
        "action_dim": int(decoder.params[-1][0].shape[-1]),
        "ppo_z_params": agent.ppo_z_state.params,
        "ppo_z_obs_stats": agent.ppo_z_state.obs_stats,
        "fm_params": decoder.params,
        "fm_obs_stats": decoder.obs_stats,
        "env_name": source_env,
        "d4rl_dataset": config.d4rl_dataset,
        "decoder_type": "fm",
        "z_dim": int(decoder.params[-1][0].shape[-1]),
        "online_config": asdict(config),
        "online_stage": stage,
        "stage_timesteps": steps,
        "best_reward": best_reward,
        "offline_config": checkpoint.get("offline_config", {}),
    }
    with open(path, "wb") as file:
        pickle.dump(payload, file)


def train_encoder(
    agent: EncoderFMAgent,
    env: Any,
    config: Config,
    stage: int,
    timesteps: int,
    metrics_path: Path,
    stage_dir: Path,
    run: Any,
    global_offset: int,
) -> tuple[EncoderFMAgent, float]:
    new_config = encoder_config(config, timesteps, env.action_size, stage)
    with jdc.copy_and_mutate(agent.ppo_z_state) as ppo:
        ppo.config = new_config
    agent = jdc.replace(agent, ppo_z_state=ppo)
    rollout = BraxRolloutState.init(
        env,
        prng=jax.random.key(config.seed + stage + 1),
        num_envs=config.num_envs,
    )
    outer_iters = timesteps // (
        new_config.iterations_per_env * new_config.num_envs
    )
    eval_iters = set(
        np.linspace(0, max(outer_iters - 1, 0), config.num_evals, dtype=int)
    )
    best_reward = -float("inf")
    best_params = agent.ppo_z_state.params
    best_stats = agent.ppo_z_state.obs_stats
    best_opt_state = agent.ppo_z_state.opt_state
    best_steps = agent.ppo_z_state.steps
    last_saved = 0
    steps_per_iter = new_config.iterations_per_env * new_config.num_envs

    for iteration in trange(outer_iters, desc=f"Stage {stage} encoder PPO"):
        completed = iteration * steps_per_iter
        if iteration in eval_iters:
            result = evaluate(
                agent,
                config,
                jax.random.fold_in(agent.ppo_z_state.prng, iteration),
            )
            append_jsonl(
                metrics_path,
                {
                    "phase": "evaluation",
                    "stage": stage,
                    "environment_steps": global_offset + completed,
                    **{f"evaluation/{k}": v for k, v in result.items()},
                },
                run,
            )
            reward = result["reward_mean"]
            if reward >= best_reward - 1e-6:
                best_reward = reward
                best_params = jax.tree.map(jnp.copy, agent.ppo_z_state.params)
                best_stats = jax.tree.map(jnp.copy, agent.ppo_z_state.obs_stats)
                best_opt_state = jax.tree.map(
                    jnp.copy, agent.ppo_z_state.opt_state
                )
                best_steps = jnp.copy(agent.ppo_z_state.steps)

        rollout, transitions = rollout.rollout(
            agent,
            episode_length=config.episode_length,
            iterations_per_env=new_config.iterations_per_env,
            apply_tanh_in_rollout=config.apply_tanh_in_rollout,
        )
        agent, metrics = agent.training_step(transitions)
        completed = (iteration + 1) * steps_per_iter
        append_jsonl(
            metrics_path,
            {
                "phase": "encoder_ppo",
                "stage": stage,
                "environment_steps": global_offset + completed,
                "encoder/reward_mean": float(np.asarray(transitions.reward).mean()),
                "encoder/z_mean": float(np.asarray(transitions.action).mean()),
                "encoder/z_std": float(np.asarray(transitions.action).std()),
                **{
                    f"ppo/{k}": float(np.asarray(v).mean())
                    for k, v in metrics.items()
                },
            },
            run,
        )
        if completed - last_saved >= config.checkpoint_interval:
            with open(
                stage_dir / f"encoder_step_{completed:012d}.pkl", "wb"
            ) as file:
                pickle.dump(
                    {
                        "ppo_z_params": agent.ppo_z_state.params,
                        "ppo_z_obs_stats": agent.ppo_z_state.obs_stats,
                        "steps": completed,
                    },
                    file,
                )
            last_saved = completed

    with jdc.copy_and_mutate(agent.ppo_z_state) as ppo:
        ppo.params = best_params
        ppo.obs_stats = best_stats
        ppo.opt_state = best_opt_state
        ppo.steps = best_steps
    return jdc.replace(agent, ppo_z_state=ppo), best_reward


def collect_data(
    agent: EncoderFMAgent,
    env: Any,
    config: Config,
    stage: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rollout = BraxRolloutState.init(
        env,
        prng=jax.random.key(config.seed + stage + 50_000),
        num_envs=config.num_envs,
    )
    states, actions, rewards = [], [], []
    for _ in trange(
        config.data_collection_iterations, desc="Collect decoder data"
    ):
        rollout, obs, act, rew = rollout.rollout_with_actions(
            agent,
            episode_length=config.episode_length,
            iterations_per_env=config.collection_steps_per_iteration,
            apply_tanh_in_rollout=config.apply_tanh_in_rollout,
        )
        states.append(np.asarray(obs).reshape(-1, obs.shape[-1]))
        actions.append(np.asarray(act).reshape(-1, act.shape[-1]))
        rewards.append(np.asarray(rew).reshape(-1))
    return (
        np.concatenate(states),
        np.concatenate(actions),
        np.concatenate(rewards),
    )

# TODO：correct selection method
def select_decoder_data(
    obs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    config: Config,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if config.fm_hybrid_sampling:
        episodes = len(obs) // config.episode_length
        usable = episodes * config.episode_length
        if episodes < 1:
            raise ValueError("Hybrid sampling requires a complete episode.")
        scores = rewards[:usable].reshape(episodes, config.episode_length).sum(1)
        threshold = np.percentile(
            scores, config.fm_high_quality_percentile * 100
        )
        high = np.flatnonzero(scores >= threshold)
        high_count = int(episodes * config.fm_high_quality_ratio)
        coverage_count = episodes - high_count
        chosen = np.concatenate(
            [
                rng.choice(high, high_count, replace=len(high) < high_count),
                rng.choice(episodes, coverage_count, replace=False),
            ]
        )
        rows = (
            chosen[:, None] * config.episode_length
            + np.arange(config.episode_length)[None]
        ).reshape(-1)
        obs, actions, rewards = obs[rows], actions[rows], rewards[rows]
    if len(obs) > config.fm_max_samples:
        rows = rng.choice(len(obs), config.fm_max_samples, replace=False)
        obs, actions, rewards = obs[rows], actions[rows], rewards[rows]
    return obs, actions, rewards


def validation_loss(
    decoder: DecoderFMState,
    obs: np.ndarray,
    actions: np.ndarray,
    key: jax.Array,
) -> tuple[float, jax.Array]:
    count = max(1, min(50, len(obs) // decoder.config.batch_size))
    losses = []
    for index in range(count):
        start = index * decoder.config.batch_size
        batch_obs = jnp.asarray(obs[start : start + decoder.config.batch_size])
        batch_act = jnp.asarray(actions[start : start + decoder.config.batch_size])
        if len(batch_obs) == 0:
            continue
        normalized = (
            batch_obs - decoder.obs_stats.mean
        ) / (decoder.obs_stats.std + 1e-8)
        key, eps_key, time_key = jax.random.split(key, 3)
        eps = jax.random.normal(
            eps_key,
            (
                len(batch_obs),
                decoder.config.n_samples_per_action,
                actions.shape[-1],
            ),
        )
        times = jax.random.uniform(
            time_key,
            (len(batch_obs), decoder.config.n_samples_per_action, 1),
        )
        losses.append(
            float(jnp.mean(decoder.compute_cfm_loss(
                normalized, batch_act, eps, times
            )))
        )
    return float(np.mean(losses)), key


def train_decoder(
    agent: EncoderFMAgent,
    obs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    config: Config,
    stage: int,
    metrics_path: Path,
    run: Any,
    environment_steps: int,
) -> EncoderFMAgent:
    rng = np.random.default_rng(config.seed + stage + 10_000)
    obs, actions, rewards = select_decoder_data(
        obs, actions, rewards, config, rng
    )
    permutation = rng.permutation(len(obs))
    val_size = max(1, int(len(obs) * config.fm_validation_fraction))
    val_rows, train_rows = permutation[:val_size], permutation[val_size:]
    if len(train_rows) < config.fm_batch_size:
        raise ValueError("Collected data is smaller than fm_batch_size.")
    decoder = agent.fm_state
    with jdc.copy_and_mutate(decoder) as decoder:
        decoder.obs_stats = decoder.obs_stats.update(jnp.asarray(obs[train_rows]))
    best_params = jax.tree.map(jnp.copy, decoder.params)
    best_opt_state = jax.tree.map(jnp.copy, decoder.opt_state)
    best_steps = jnp.copy(decoder.steps)
    best_loss, stale = float("inf"), 0
    key = jax.random.key(config.seed + stage + 70_000)

    for epoch in trange(config.fm_num_epochs, desc=f"Stage {stage} FM"):
        losses = []
        shuffled = rng.permutation(train_rows)
        for start in range(0, len(shuffled) - config.fm_batch_size + 1,
                           config.fm_batch_size):
            rows = shuffled[start : start + config.fm_batch_size]
            decoder, metrics = decoder.train_step(
                jnp.asarray(obs[rows]), jnp.asarray(actions[rows])
            )
            losses.append(float(metrics["loss"]))
        val_loss, key = validation_loss(
            decoder, obs[val_rows], actions[val_rows], key
        )
        append_jsonl(
            metrics_path,
            {
                "phase": "decoder_fm",
                "stage": stage,
                "environment_steps": environment_steps,
                "decoder/epoch": epoch + 1,
                "decoder/train_loss": float(np.mean(losses)),
                "decoder/validation_loss": val_loss,
            },
            run,
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = jax.tree.map(jnp.copy, decoder.params)
            best_opt_state = jax.tree.map(jnp.copy, decoder.opt_state)
            best_steps = jnp.copy(decoder.steps)
            stale = 0
        else:
            stale += 1
            if stale >= config.fm_patience:
                break
    with jdc.copy_and_mutate(decoder) as decoder:
        decoder.params = best_params
        decoder.opt_state = best_opt_state
        decoder.steps = best_steps
    return jdc.replace(agent, fm_state=decoder)


def main(config: Config) -> None:
    validate_config(config)
    timesteps = parse_timesteps(config)
    if config.offline_checkpoint is None:
        header: dict[str, Any] = {}
    else:
        with open(
            Path(config.offline_checkpoint).expanduser(), "rb"
        ) as file:
            header = pickle.load(file)
    source_env, task = resolve_task(header, config)
    env = brax_envs.get_environment(task, backend="mjx")
    checkpoint, encoder, decoder = load_states(
        config.offline_checkpoint, env, config, timesteps[0]
    )
    agent = EncoderFMAgent(ppo_z_state=encoder, fm_state=decoder)

    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    with open(output / "config.json", "w") as file:
        json.dump(
            {**asdict(config), "resolved_environment": source_env,
             "mjx_task": task},
            file,
            indent=2,
        )

    run = None
    if config.wandb_enabled:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "wandb is required unless --no-wandb-enabled is passed."
            ) from error
        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name or output.name,
            group=config.wandb_group,
            tags=list(config.wandb_tags),
            mode=config.wandb_mode,
            config=asdict(config),
        )

    offset = 0
    best_reward = -float("inf")
    try:
        for stage, stage_steps in enumerate(timesteps):
            stage_dir = output / f"stage_{stage}"
            stage_dir.mkdir(parents=True, exist_ok=True)
            agent, stage_reward = train_encoder(
                agent, env, config, stage, stage_steps, metrics_path,
                stage_dir, run, offset
            )
            best_reward = max(best_reward, stage_reward)
            obs, actions, rewards = collect_data(agent, env, config, stage)
            with open(stage_dir / "online_decoder_data.pkl", "wb") as file:
                pickle.dump(
                    {
                        "states": obs,
                        "actions": actions,
                        "rewards": rewards,
                        "env_name": source_env,
                        "d4rl_dataset": config.d4rl_dataset,
                    },
                    file,
                )
            agent = train_decoder(
                agent, obs, actions, rewards, config, stage, metrics_path,
                run, offset + stage_steps
            )
            save_checkpoint(
                stage_dir / "checkpoint_final.pkl",
                checkpoint,
                config,
                source_env,
                stage,
                stage_steps,
                agent,
                best_reward,
            )
            offset += stage_steps
        save_checkpoint(
            output / "checkpoint_final.pkl",
            checkpoint,
            config,
            source_env,
            config.num_stages - 1,
            timesteps[-1],
            agent,
            best_reward,
        )
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
