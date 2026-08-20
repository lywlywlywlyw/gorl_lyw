"""Train an RLPD (high-UTD SAC) encoder in a frozen decoder's latent space."""

from __future__ import annotations

import datetime
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import tyro
from tqdm import tqdm

from envs.robomimic.RobomimicEnv import RobomimicEnv
from envs.robomimic.online_config.env_config import EnvConfig
from envs.robomimic.online_config.encoder_configs.rlpd_config import RLPDConfig
from envs.robomimic.online_config.training_config import TrainingConfig
from flow_policy import encoder_rlpd
from flow_policy.agent import EncoderFMAgent
from flow_policy.decoder_fm import DecoderFMState
from flow_policy.rollout_encoder import (
    BatchedRolloutStateEncoderFM,
    eval_policy_encoder_fm,
)
from metrics_ipc import append_metrics


@dataclass
class ReplayBuffer:
    observations: np.ndarray
    latents: np.ndarray
    env_actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    masks: np.ndarray
    is_new: np.ndarray
    capacity: int
    size: int = 0
    cursor: int = 0

    @classmethod
    def create(cls, capacity: int, obs_dim: int, action_dim: int) -> "ReplayBuffer":
        return cls(
            observations=np.empty((capacity, obs_dim), dtype=np.float32),
            latents=np.empty((capacity, action_dim), dtype=np.float32),
            env_actions=np.empty((capacity, action_dim), dtype=np.float32),
            rewards=np.empty((capacity,), dtype=np.float32),
            next_observations=np.empty((capacity, obs_dim), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            is_new=np.empty((capacity,), dtype=np.bool_),
            capacity=capacity,
        )

    def add_rollout(self, transitions: Any) -> None:
        arrays = {
            "observations": np.asarray(jax.device_get(transitions.obs)).reshape(
                -1, self.observations.shape[-1]
            ),
            "latents": np.asarray(jax.device_get(transitions.action)).reshape(
                -1, self.latents.shape[-1]
            ),
            "env_actions": np.asarray(
                jax.device_get(transitions.action_info["env_action"])
            ).reshape(-1, self.env_actions.shape[-1]),
            "rewards": np.asarray(jax.device_get(transitions.reward)).reshape(-1),
            "next_observations": np.asarray(
                jax.device_get(transitions.next_obs)
            ).reshape(-1, self.next_observations.shape[-1]),
            "masks": np.asarray(jax.device_get(transitions.discount)).reshape(-1),
            "is_new": np.ones(
                np.asarray(jax.device_get(transitions.reward)).size, dtype=np.bool_
            ),
        }
        count = len(arrays["rewards"])
        if count >= self.capacity:
            arrays = {key: value[-self.capacity :] for key, value in arrays.items()}
            count = self.capacity
        indices = (np.arange(count) + self.cursor) % self.capacity
        for key, value in arrays.items():
            getattr(self, key)[indices] = value
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, rng: np.random.Generator, batch_size: int) -> dict[str, np.ndarray]:
        if self.size == 0:
            raise ValueError("Cannot sample an empty replay buffer.")
        indices = rng.integers(0, self.size, size=batch_size)
        return {
            key: getattr(self, key)[indices]
            for key in (
                "observations",
                "latents",
                "env_actions",
                "rewards",
                "next_observations",
                "masks",
                "is_new",
            )
        }

    def mark_all_old(self) -> None:
        """Invalidate rollout latents after the current encoder stage."""
        self.is_new[: self.size] = False

    def save(self, path: str | None) -> None:
        if path is None:
            return
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load_or_create(
        cls, path: str | None, capacity: int, obs_dim: int, action_dim: int
    ) -> "ReplayBuffer":
        source = None if path is None else Path(path).expanduser()
        if source is not None and source.is_file():
            with source.open("rb") as file:
                replay = pickle.load(file)
            if not isinstance(replay, cls):
                raise TypeError(f"Invalid encoder replay buffer: {source}")
            return replay
        return cls.create(capacity, obs_dim, action_dim)


def _load_encoder_demo_buffer(path: str | None, obs_dim: int, action_dim: int):
    if path is None:
        return None
    dataset_path = Path(path).expanduser()
    with open(dataset_path, "rb") as file:
        data = pickle.load(file)
    if not isinstance(data, dict):
        raise ValueError("Encoder demo buffer must be a pickle dictionary.")
    aliases = {
        "observations": ("observations", "states", "obs"),
        "env_actions": ("env_actions", "actions"),
        "rewards": ("rewards", "reward"),
        "next_observations": ("next_observations", "next_states", "next_obs"),
        "masks": ("masks", "discounts", "discount"),
    }
    arrays = {}
    for target, candidates in aliases.items():
        value = next((data[key] for key in candidates if key in data), None)
        if value is None:
            raise KeyError(f"Encoder demo buffer is missing {target}.")
        arrays[target] = np.asarray(value, dtype=np.float32)
    if arrays["observations"].shape[-1] != obs_dim:
        raise ValueError("Offline dataset observation dimension does not match env.")
    if arrays["env_actions"].shape[-1] != action_dim:
        raise ValueError("Encoder demo actions do not match the environment action dimension.")
    size = len(arrays["rewards"])
    if not all(len(value) == size for value in arrays.values()):
        raise ValueError("Offline latent transition arrays have unequal lengths.")
    return arrays


def _inverse_fm_batch(
    decoder: DecoderFMState,
    observations: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    obs = jnp.asarray(observations)
    x = jnp.asarray(actions)
    obs_norm = (obs - decoder.obs_stats.mean) / (decoder.obs_stats.std + 1e-8)
    # times = jnp.linspace(1.0, 0.0, decoder.config.flow_steps + 1)
    times = jnp.linspace(0.0, 1.0, decoder.config.flow_steps + 1)

    def step(x_t, pair):
        current, following = pair
        t = jnp.full((*x_t.shape[:-1], 1), current)
        velocity = decoder.flow_forward(obs_norm, x_t, decoder.embed_timestep(t))
        return x_t + (following - current) * velocity, None

    latent, _ = jax.lax.scan(step, x, (times[:-1], times[1:]))
    return np.asarray(jax.device_get(latent), dtype=np.float32)


def _sample_mixed_batch(
    online: ReplayBuffer,
    demo: dict[str, np.ndarray] | None,
    demo_ratio: float,
    batch_size: int,
    rng: np.random.Generator,
    decoder: DecoderFMState,
) -> encoder_rlpd.RLPDTransitionBatch:
    demo_count = 0 if demo is None else int(round(batch_size * demo_ratio))
    demo_count = min(batch_size, max(0, demo_count))
    online_count = batch_size - demo_count
    if online_count and online.size == 0:
        demo_count, online_count = batch_size, 0
    pieces: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "masks",
        )
    }
    if online_count:
        sampled = online.sample(rng, online_count)
        latents = sampled["latents"].copy()
        old = ~sampled["is_new"]
        if np.any(old):
            latents[old] = _inverse_fm_batch(
                decoder, sampled["observations"][old], sampled["env_actions"][old]
            )
        pieces["observations"].append(sampled["observations"])
        pieces["actions"].append(latents)
        for key in ("rewards", "next_observations", "masks"):
            pieces[key].append(sampled[key])
    if demo_count:
        if demo is None:
            raise ValueError("Demo samples requested without an encoder demo buffer.")
        indices = rng.integers(0, len(demo["rewards"]), size=demo_count)
        demo_obs = demo["observations"][indices]
        pieces["observations"].append(demo_obs)
        pieces["actions"].append(
            _inverse_fm_batch(decoder, demo_obs, demo["env_actions"][indices])
        )
        for key in ("rewards", "next_observations", "masks"):
            pieces[key].append(demo[key][indices])
    batch = {
        key: jnp.asarray(np.concatenate(value, axis=0))
        for key, value in pieces.items()
    }
    return encoder_rlpd.RLPDTransitionBatch(**batch)


def _checkpoint(
    agent: EncoderFMAgent,
    encoder_config: encoder_rlpd.EncoderConfig,
    config: dict[str, Any],
    iteration: int,
    best_reward: float,
    z_dim: int,
) -> dict[str, Any]:
    state = agent.ppo_z_state
    return {
        "rlpd_z_actor_params": state.actor_params,
        "rlpd_z_critic_params": state.critic_params,
        "rlpd_z_target_critic_params": state.target_critic_params,
        "rlpd_z_log_temperature": state.log_temperature,
        "rlpd_z_actor_opt_state": state.actor_opt_state,
        "rlpd_z_critic_opt_state": state.critic_opt_state,
        "rlpd_z_temperature_opt_state": state.temperature_opt_state,
        "rlpd_z_obs_stats": state.obs_stats,
        "rlpd_z_prng": state.prng,
        "rlpd_z_steps": state.steps,
        "fm_params": agent.fm_state.params,
        "fm_obs_stats": agent.fm_state.obs_stats,
        "config": encoder_config,
        "env_name": config["env_name"],
        "decoder_type": "fm",
        "final_iteration": iteration,
        "best_reward": best_reward,
        "z_dim": z_dim,
    }


def _tree_shapes(tree: Any) -> Any:
    return jax.tree.map(lambda value: tuple(value.shape), tree)


def _validate_iql_actor_warm_start(
    encoder_state: encoder_rlpd.EncoderState,
    checkpoint: dict[str, Any],
    checkpoint_path: str,
) -> None:
    """Reject incompatible legacy IQL actors with an actionable error."""
    expected = _tree_shapes(encoder_state.actor_params)
    actual = _tree_shapes(checkpoint["iql_z_actor_params"])
    same_structure = jax.tree.structure(expected) == jax.tree.structure(actual)
    same_shapes = same_structure and all(
        left == right
        for left, right in zip(
            jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
        )
    )
    if not same_shapes:
        raise ValueError(
            "Offline IQL actor architecture does not match the online RLPD actor. "
            f"Checkpoint: {checkpoint_path}. Re-run "
            "run_offline_fm_frozen_robomimic.py with the current code/config."
        )


def main(
    exp_name: str,
    decoder_model_path: str | None = None,
    num_timesteps: int | None = None,
    stage: int = 0,
    global_step_offset: int = 0,
    metrics_file: str | None = None,
    stage_init_before_training: bool = True,
    replay_buffer_path: str | None = None,
) -> None:
    config = TrainingConfig().to_dict() | EnvConfig().to_dict() | RLPDConfig().to_dict()
    if config["decoder_type"] != "fm":
        raise ValueError("train_encoder_rlpd.py currently supports the FM decoder only.")
    total_timesteps = config["encoder_num_timesteps"] if num_timesteps is None else num_timesteps
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / f"encoder_rlpd_fm_{config['env_name']}_{exp_name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    if decoder_model_path is None:
        candidates = sorted(Path("fm_models").glob("fm_model_best_*.pkl"))
        if not candidates:
            raise ValueError("No FM model found; specify --decoder-model-path.")
        decoder_model_path = str(candidates[-1])
    with open(decoder_model_path, "rb") as file:
        decoder_checkpoint = pickle.load(file)
    required = {"params", "obs_stats", "config", "obs_dim", "action_dim"}
    missing = sorted(required.difference(decoder_checkpoint))
    if missing:
        raise ValueError(f"Decoder checkpoint is missing fields: {missing}")

    env = RobomimicEnv(
        dataset_path=config["dataset_path"], reward_shaping=config["dense_reward"]
    )
    z_dim = int(env.action_size)
    encoder_config = encoder_rlpd.EncoderConfig(
        learning_rate=config["rlpd_actor_learning_rate"],
        critic_learning_rate=config["rlpd_critic_learning_rate"],
        temperature_learning_rate=config["rlpd_temperature_learning_rate"],
        discounting=config["rlpd_discounting"],
        episode_length=config["episode_length"],
        normalize_observations=config["rlpd_normalize_observations"],
        num_envs=config["num_envs"],
        z_dim=z_dim,
        hidden_size=config["rlpd_hidden_size"],
        hidden_layers=config["rlpd_hidden_layers"],
        critic_ensemble_size=config["rlpd_critic_ensemble_size"],
        critic_subsample_size=config["rlpd_critic_subsample_size"],
        target_update_rate=config["rlpd_target_update_rate"],
        initial_temperature=config["rlpd_initial_temperature"],
        target_entropy=config["rlpd_target_entropy"],
        backup_entropy=config["rlpd_backup_entropy"],
        reward_scaling=config["rlpd_reward_scaling"],
        reward_bias=config["rlpd_reward_bias"],
        max_grad_norm=config["rlpd_max_grad_norm"],
        policy_update_period=config["rlpd_policy_update_period"],
        apply_tanh_in_rollout=config["rlpd_apply_tanh_in_rollout"],
    )
    encoder_state = encoder_rlpd.EncoderState.init(
        jax.random.key(config["seed"]), env, encoder_config
    )
    has_encoder_state = "rlpd_z_actor_params" in decoder_checkpoint
    has_iql_actor_warm_start = "iql_z_actor_params" in decoder_checkpoint
    should_resume_encoder = not (
        stage == 0 and decoder_checkpoint.get("is_identity", False)
    )
    if not stage_init_before_training and should_resume_encoder and not has_encoder_state:
        raise ValueError(
            "stage_init_before_training=False requires previous RLPD encoder state "
            f"for stage {stage}; checkpoint {decoder_model_path} does not contain "
            "'rlpd_z_actor_params'."
        )
    resume = (
        not stage_init_before_training
        and should_resume_encoder
        and has_encoder_state
    )
    decoder_checkpoint_is_offline = bool(
        decoder_checkpoint.get("is_frozen_offline", False)
        or decoder_checkpoint.get("offline_checkpoint_type") is not None
        or str(decoder_checkpoint.get("checkpoint_format", "")).startswith(
            "gorl_offline"
        )
    )
    if resume:
        with jdc.copy_and_mutate(encoder_state) as state:
            state.actor_params = decoder_checkpoint["rlpd_z_actor_params"]
            state.critic_params = decoder_checkpoint["rlpd_z_critic_params"]
            state.target_critic_params = decoder_checkpoint["rlpd_z_target_critic_params"]
            state.log_temperature = decoder_checkpoint["rlpd_z_log_temperature"]
            state.obs_stats = decoder_checkpoint["rlpd_z_obs_stats"]
            # Never inherit an offline optimizer. Online-to-online continuation
            # may preserve optimizer/PRNG/step state as before.
            if not decoder_checkpoint_is_offline:
                for name in (
                    "actor_opt_state",
                    "critic_opt_state",
                    "temperature_opt_state",
                    "prng",
                    "steps",
                ):
                    key = f"rlpd_z_{name}"
                    if key in decoder_checkpoint:
                        setattr(state, name, decoder_checkpoint[key])
    elif has_iql_actor_warm_start:
        _validate_iql_actor_warm_start(
            encoder_state, decoder_checkpoint, decoder_model_path
        )
        with jdc.copy_and_mutate(encoder_state) as state:
            state.actor_params = decoder_checkpoint["iql_z_actor_params"]
            state.obs_stats = decoder_checkpoint.get(
                "iql_z_obs_stats", state.obs_stats
            )
        print(
            "Warm-started online RLPD actor and observation statistics from "
            f"offline IQL checkpoint: {decoder_model_path}"
        )

    decoder_state = DecoderFMState.init(
        jax.random.PRNGKey(config["seed"] + 1000),
        decoder_checkpoint["obs_dim"],
        decoder_checkpoint["action_dim"],
        decoder_checkpoint["config"],
    )
    with jdc.copy_and_mutate(decoder_state) as state:
        state.params = decoder_checkpoint["params"]
        state.obs_stats = decoder_checkpoint["obs_stats"]
    agent = EncoderFMAgent(ppo_z_state=encoder_state, fm_state=decoder_state)
    rollout_state = BatchedRolloutStateEncoderFM.init(
        env, jax.random.key(config["seed"] + 1), config["num_envs"]
    )
    replay = ReplayBuffer.load_or_create(
        replay_buffer_path,
        config["rlpd_replay_buffer_capacity"],
        int(env.observation_size),
        z_dim,
    )
    demo = _load_encoder_demo_buffer(
        config["rlpd_offline_latent_dataset_path"], int(env.observation_size), z_dim
    )
    np_rng = np.random.default_rng(config["seed"])
    rollout_steps = config["rlpd_rollout_steps_per_iteration"]
    steps_per_iteration = rollout_steps * config["num_envs"]
    outer_iters = max(1, int(np.ceil(total_timesteps / steps_per_iteration)))
    eval_iters = set(np.linspace(0, outer_iters - 1, config["rlpd_num_evals"], dtype=int))
    best_reward = -float("inf")
    update_count = 0
    started = time.time()

    for iteration in tqdm(range(outer_iters), desc="RLPD encoder"):
        rollout_state, transitions = rollout_state.rollout(
            agent,
            episode_length=config["episode_length"],
            iterations_per_env=rollout_steps,
            apply_tanh_in_rollout=config["rlpd_apply_tanh_in_rollout"],
        )
        replay.add_rollout(transitions)
        observations = jnp.concatenate(
            [transitions.obs.reshape(-1, int(env.observation_size)),
             transitions.next_obs.reshape(-1, int(env.observation_size))],
            axis=0,
        )
        agent = jdc.replace(
            agent, ppo_z_state=agent.ppo_z_state.update_observation_stats(observations)
        )
        metrics: dict[str, Any] = {}
        env_steps = (iteration + 1) * steps_per_iteration
        if replay.size >= config["rlpd_learning_starts"] or demo is not None:
            updates = config["rlpd_updates_per_env_step"] * steps_per_iteration
            for _ in range(updates):
                batch = _sample_mixed_batch(
                    replay,
                    demo,
                    config["rlpd_offline_ratio"] if demo is not None else 0.0,
                    config["rlpd_batch_size"],
                    np_rng,
                    agent.fm_state,
                )
                encoder_state, critic_metrics = agent.ppo_z_state.update_critic(batch)
                update_count += 1
                metrics = dict(critic_metrics)
                if update_count % config["rlpd_policy_update_period"] == 0:
                    encoder_state, actor_metrics = encoder_state.update_actor_and_temperature(batch)
                    metrics.update(actor_metrics)
                agent = jdc.replace(agent, ppo_z_state=encoder_state)

        global_step = global_step_offset + min(env_steps, total_timesteps)
        if iteration in eval_iters:
            evaluation = eval_policy_encoder_fm(
                agent,
                prng=jax.random.fold_in(agent.ppo_z_state.prng, iteration),
                num_envs=config["eval_num_envs"],
                max_episode_length=config["episode_length"],
                apply_tanh_in_rollout=config["rlpd_apply_tanh_in_rollout"],
            )
            evaluation_metrics = {
                key: float(np.asarray(value))
                for key, value in evaluation.scalar_metrics.items()
            }
            reward = evaluation_metrics.get("reward_mean", -float("inf"))
            log = {
                "pipeline/env_step": global_step,
                "pipeline/stage": stage,
                "replay/size": replay.size,
                "train/updates": update_count,
                **{f"train/{key}": float(np.asarray(value)) for key, value in metrics.items()},
                **{f"eval/{key}": value for key, value in evaluation_metrics.items()},
            }
            if metrics_file is not None:
                append_metrics(metrics_file, log)
            if reward > best_reward:
                best_reward = reward
                with open(results_dir / "best_checkpoint.pkl", "wb") as file:
                    pickle.dump(
                        _checkpoint(agent, encoder_config, config, iteration + 1, best_reward, z_dim),
                        file,
                    )
        if (iteration + 1) % config["rlpd_checkpoint_interval"] == 0:
            with open(results_dir / f"checkpoint_{iteration + 1}.pkl", "wb") as file:
                pickle.dump(
                    _checkpoint(agent, encoder_config, config, iteration + 1, best_reward, z_dim),
                    file,
                )

    replay.mark_all_old()
    replay.save(replay_buffer_path)
    with open(results_dir / "final_checkpoint.pkl", "wb") as file:
        pickle.dump(
            _checkpoint(agent, encoder_config, config, outer_iters, best_reward, z_dim), file
        )
    completion = {
        "pipeline/env_step": global_step_offset + total_timesteps,
        "pipeline/stage": stage,
        "stage/best_reward": best_reward,
        "stage/encoder_completed": 1,
        f"stage_{stage}/encoder_completed": 1,
        "train/wall_time_seconds": time.time() - started,
    }
    if metrics_file is not None:
        append_metrics(metrics_file, completion)
    print(f"RLPD encoder results saved to {results_dir}")


if __name__ == "__main__":
    tyro.cli(main)
