"""MJX-parallel environment frontend for ``run_gorl_fm_new.py``.

The GoRL algorithm, network definitions, checkpoint format, and all training
defaults are imported unchanged from ``run_gorl_fm_new.py``.  This file only
replaces Gymnasium ``SyncVectorEnv`` execution with Brax environments using
the MJX backend and JAX-vmapped environment stepping.

The Brax locomotion tasks are MJX counterparts of the D4RL MuJoCo task
families, not byte-for-byte reproductions of Gymnasium's environment classes.
Checkpoint observation and action dimensions are validated by the shared
training code before training starts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import jax
import numpy as np
import tyro
from brax import envs as brax_envs
from brax.envs.wrappers import training as brax_training
from jax import numpy as jnp

import run_gorl_fm_new as shared
import datetime

# Inherit every algorithm/training/network default from the shared version and
# add only experiment-tracking options.
@dataclass
class OnlineConfig(shared.OnlineConfig):
    wandb_enabled: bool = True
    wandb_project: str = "GoRL-online"
    wandb_entity: str | None = None
    wandb_run_name: str | None = "gorl_onine_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_mode: str = "online"
    wandb_group: str | None = None
    wandb_tags: tuple[str, ...] = ()


RolloutBatch = shared.RolloutBatch


def _mjx_task_name(env_id: str) -> str:
    """Maps a D4RL/Gymnasium task identifier to its Brax MJX task."""
    family = env_id.split("-", 1)[0].lower()
    aliases = {
        "walker2d": "walker2d",
        "hopper": "hopper",
        "halfcheetah": "halfcheetah",
        "ant": "ant",
    }
    if family not in aliases:
        raise ValueError(
            f"Unsupported MJX environment {env_id!r}; supported D4RL "
            f"families are {sorted(aliases)}."
        )
    return aliases[family]


class MjxVectorEnv:
    """Small Gymnasium-vector-compatible facade over a vmapped MJX task."""

    def __init__(
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        episode_length: int,
    ):
        self.env_id = env_id
        self.num_envs = num_envs
        self.seed = seed
        self.episode_length = episode_length
        task_name = _mjx_task_name(env_id)
        base_env = brax_envs.get_environment(task_name, backend="mjx")
        self._env = brax_training.wrap(
            base_env,
            episode_length=episode_length,
            action_repeat=1,
        )
        self._reset = jax.jit(self._env.reset)
        self._step = jax.jit(self._env.step)
        self._state: Any | None = None
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(base_env.observation_size),),
            dtype=np.float32,
        )
        self.single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(int(base_env.action_size),),
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: Any | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        reset_seed = self.seed if seed is None else seed
        # Brax 0.14 training wrappers inspect the trailing legacy key axis to
        # infer batch dimensions, so use PRNGKey rather than typed key().
        keys = jax.random.split(
            jax.random.PRNGKey(reset_seed), self.num_envs
        )
        self._state = self._reset(keys)
        return np.asarray(self._state.obs, dtype=np.float32), {}

    def step(
        self, actions: np.ndarray
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, Any],
    ]:
        if self._state is None:
            raise RuntimeError("MJX environment must be reset before step().")
        self._state = self._step(
            self._state, jnp.asarray(actions, dtype=jnp.float32)
        )
        done = np.asarray(self._state.done, dtype=bool)
        truncation = np.asarray(
            self._state.info.get(
                "truncation", jnp.zeros_like(self._state.done)
            ),
            dtype=bool,
        )
        terminated = np.logical_and(done, np.logical_not(truncation))
        return (
            np.asarray(self._state.obs, dtype=np.float32),
            np.asarray(self._state.reward, dtype=np.float32),
            terminated,
            truncation,
            {},
        )

    def close(self) -> None:
        self._state = None


_episode_length = 1000
_wandb_run: Any | None = None
_stage_offsets: tuple[int, ...] = ()
_stage_lengths: tuple[int, ...] = ()
_jsonl_writer = shared.append_jsonl


def make_vector_env(env_id: str, num_envs: int, seed: int) -> MjxVectorEnv:
    return MjxVectorEnv(
        env_id,
        num_envs,
        seed,
        episode_length=_episode_length,
    )


def evaluate_policy(
    env_id: str,
    params: Any,
    obs_stats: Any,
    decoder: Any,
    episodes: int,
    seed: int,
    apply_tanh: bool,
) -> dict[str, float]:
    """Runs deterministic evaluation episodes in one MJX-vmapped batch."""
    env = MjxVectorEnv(
        env_id,
        episodes,
        seed,
        episode_length=_episode_length,
    )
    try:
        observations, _ = env.reset(seed=seed)
        returns = np.zeros(episodes, dtype=np.float64)
        lengths = np.zeros(episodes, dtype=np.int32)
        active = np.ones(episodes, dtype=bool)
        while np.any(active):
            distribution, _ = shared.policy_value(
                params, obs_stats, jnp.asarray(observations)
            )
            decoded = np.asarray(
                shared.decode_actions(
                    decoder, jnp.asarray(observations), distribution.loc
                )
            )
            actions = np.tanh(decoded) if apply_tanh else decoded
            actions = np.clip(
                actions,
                env.single_action_space.low,
                env.single_action_space.high,
            )
            observations, rewards, terminated, truncated, _ = env.step(
                actions
            )
            returns[active] += rewards[active]
            lengths[active] += 1
            active &= np.logical_not(
                np.logical_or(terminated, truncated)
            )
        return {
            "reward_mean": float(np.mean(returns)),
            "reward_std": float(np.std(returns)),
            "reward_min": float(np.min(returns)),
            "reward_max": float(np.max(returns)),
            "episode_length_mean": float(np.mean(lengths)),
        }
    finally:
        env.close()


def _wandb_append_jsonl(
    path: Path, record: dict[str, Any]
) -> None:
    """Preserves JSONL logging and mirrors useful metrics to W&B."""
    _jsonl_writer(path, record)
    if _wandb_run is None:
        return

    phase = str(record.get("phase", "training"))
    stage = int(record.get("stage", 0))
    if phase == "decoder_fm" and "stage_timesteps" not in record:
        stage_steps = (
            _stage_lengths[stage] if stage < len(_stage_lengths) else 0
        )
    else:
        stage_steps = int(record.get("stage_timesteps", 0))
    stage_offset = (
        _stage_offsets[stage] if stage < len(_stage_offsets) else 0
    )
    payload: dict[str, Any] = {
        "stage": stage,
        "environment_steps": stage_offset + stage_steps,
    }
    for name, value in record.items():
        if name in {"phase", "stage", "stage_timesteps"}:
            continue
        if name.startswith("ppo/"):
            metric_name = name
        elif phase == "encoder_ppo":
            metric_name = f"encoder/{name}"
        elif phase == "evaluation":
            metric_name = f"evaluation/{name}"
        elif phase == "decoder_fm":
            metric_name = f"decoder/{name}"
        else:
            metric_name = f"{phase}/{name}"
        payload[metric_name] = value
    if phase == "decoder_fm":
        payload["decoder/epoch"] = int(record.get("epoch", 0))
    _wandb_run.log(payload)


def main(config: OnlineConfig) -> None:
    """Runs the shared GoRL implementation with MJX environment hooks."""
    global _episode_length, _stage_offsets, _stage_lengths, _wandb_run
    _episode_length = config.episode_length
    stage_timesteps = shared.parse_stage_timesteps(config)
    offsets = []
    cumulative = 0
    for timesteps in stage_timesteps:
        offsets.append(cumulative)
        cumulative += timesteps
    _stage_offsets = tuple(offsets)
    _stage_lengths = tuple(stage_timesteps)

    shared.make_vector_env = make_vector_env
    shared.evaluate_policy = evaluate_policy
    shared.append_jsonl = _wandb_append_jsonl

    if config.wandb_enabled:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "W&B logging is enabled but wandb is not installed. "
                "Install wandb or pass --no-wandb-enabled."
            ) from error
        run_name = config.wandb_run_name or Path(config.output_dir).name
        _wandb_run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=run_name,
            group=config.wandb_group,
            tags=list(config.wandb_tags),
            mode=config.wandb_mode,
            config=asdict(config),
        )
        wandb.define_metric("environment_steps")
        wandb.define_metric(
            "encoder/*", step_metric="environment_steps"
        )
        wandb.define_metric(
            "evaluation/*", step_metric="environment_steps"
        )
        wandb.define_metric("ppo/*", step_metric="environment_steps")
        wandb.define_metric(
            "decoder/*", step_metric="environment_steps"
        )

    try:
        shared.main(config)
    finally:
        shared.append_jsonl = _jsonl_writer
        if _wandb_run is not None:
            _wandb_run.finish()
            _wandb_run = None


if __name__ == "__main__":
    main(tyro.cli(OnlineConfig))
