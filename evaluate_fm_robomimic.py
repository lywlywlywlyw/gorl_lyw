"""Evaluate checkpoints produced by ``run_offline_fm_frozen_new.py``.

The evaluator uses the same local D4RL MJX environment, latent Gaussian policy,
observation normalizer, and multi-step FM decoder as offline training. Both
``checkpoint_final.pkl`` and periodic ``checkpoint_step_*.pkl`` files work.

Example:
    python evaluate_offline_fm.py \
        --checkpoint results/.../checkpoint_final.pkl \
        --episodes 50 --deterministic
"""

from __future__ import annotations

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
from jax import numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from d4rl_envs.mjx_envs import make_d4rl_env, normalize_task
from flow_policy import networks
from flow_policy.decoder_fm import DecoderFMState


# D4RL v2 reference returns used by get_normalized_score().
D4RL_SCORE_RANGES: dict[str, tuple[float, float]] = {
    "halfcheetah": (-280.178953, 12135.0),
    "hopper": (-20.272305, 3234.3),
    "walker2d": (1.629008, 4592.3),
    "ant": (-325.6, 3879.7),
}


@dataclass
class EvaluationConfig:
    """Command-line configuration for frozen offline-FM evaluation."""

    checkpoint: str
    env_name: str | None = None
    d4rl_dataset: str | None = None
    episodes: int = 50
    episode_length: int | None = None
    seed: int = 0
    deterministic: bool = True
    clip_actions: bool = True
    output_json: str | None = None


@dataclass(frozen=True)
class LoadedPolicy:
    env: Any
    task: str
    dataset_name: str
    actor_params: Any
    actor_obs_stats: Any
    decoder: DecoderFMState
    normalize_actor_observations: bool
    episode_length: int


def _load_checkpoint(path: str) -> tuple[Path, dict[str, Any]]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    with checkpoint_path.open("rb") as file:
        checkpoint = pickle.load(file)
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected a dictionary checkpoint, got {type(checkpoint).__name__}."
        )
    return checkpoint_path, checkpoint


def _config_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        return asdict(value)
    except (TypeError, ValueError):
        return vars(value) if hasattr(value, "__dict__") else {}


def _resolve_environment(
    checkpoint: dict[str, Any], config: EvaluationConfig
) -> tuple[str, str]:
    offline_config = _config_dict(checkpoint.get("offline_config", {}))
    source = (
        config.env_name
        or config.d4rl_dataset
        or checkpoint.get("env_name")
        or checkpoint.get("d4rl_dataset")
        or offline_config.get("env_name")
        or offline_config.get("d4rl_dataset")
    )
    if source is None:
        raise ValueError(
            "Could not infer the environment from the checkpoint. Pass "
            "--env-name or --d4rl-dataset."
        )
    return normalize_task(str(source)), str(source)


def _actor_policy_params(checkpoint: dict[str, Any]) -> Any:
    params = checkpoint.get("ppo_z_params")
    if params is None:
        raise KeyError(
            "Checkpoint is missing 'ppo_z_params'; expected a checkpoint from "
            "run_offline_fm_frozen_new.py."
        )
    if hasattr(params, "policy"):
        return params.policy
    if isinstance(params, dict) and "policy" in params:
        return params["policy"]
    raise TypeError("'ppo_z_params' does not contain policy parameters.")


def _decoder_fields(
    checkpoint: dict[str, Any],
) -> tuple[Any, Any, Any, int, int]:
    params = checkpoint.get("fm_params", checkpoint.get("params"))
    obs_stats = checkpoint.get("fm_obs_stats", checkpoint.get("obs_stats"))
    decoder_config = checkpoint.get("config")
    obs_dim = checkpoint.get("obs_dim")
    action_dim = checkpoint.get("action_dim", checkpoint.get("z_dim"))

    missing = [
        name
        for name, value in (
            ("fm_params/params", params),
            ("fm_obs_stats/obs_stats", obs_stats),
            ("config", decoder_config),
            ("obs_dim", obs_dim),
            ("action_dim/z_dim", action_dim),
        )
        if value is None
    ]
    if missing:
        raise KeyError("Checkpoint is missing fields: " + ", ".join(missing))
    if not hasattr(decoder_config, "flow_steps"):
        raise TypeError(
            "Checkpoint 'config' is not a DecoderFMConfig. Evaluate "
            "checkpoint_final.pkl or checkpoint_step_*.pkl, not the separate "
            "encoder_checkpoint_*.pkl file."
        )
    return params, obs_stats, decoder_config, int(obs_dim), int(action_dim)


def load_policy(config: EvaluationConfig) -> tuple[Path, LoadedPolicy]:
    checkpoint_path, checkpoint = _load_checkpoint(config.checkpoint)
    task, dataset_name = _resolve_environment(checkpoint, config)
    env = make_d4rl_env(task)
    decoder_params, decoder_obs_stats, decoder_config, obs_dim, action_dim = (
        _decoder_fields(checkpoint)
    )

    if env.observation_size != obs_dim:
        raise ValueError(
            f"Checkpoint observation dimension {obs_dim} does not match "
            f"{task} environment dimension {env.observation_size}."
        )
    if env.action_size != action_dim:
        raise ValueError(
            f"Checkpoint action dimension {action_dim} does not match "
            f"{task} environment dimension {env.action_size}."
        )

    decoder = DecoderFMState.init(
        jax.random.key(config.seed + 1), obs_dim, action_dim, decoder_config
    )
    with jdc.copy_and_mutate(decoder) as decoder:
        decoder.params = decoder_params
        decoder.obs_stats = decoder_obs_stats

    actor_obs_stats = checkpoint.get("ppo_z_obs_stats")
    if actor_obs_stats is None:
        raise KeyError(
            "Checkpoint is missing 'ppo_z_obs_stats', which is required to "
            "reproduce the actor observation normalization used in training."
        )

    offline_config = _config_dict(checkpoint.get("offline_config", {}))
    online_config = checkpoint.get("online_encoder_config")
    normalize_actor_observations = bool(
        getattr(online_config, "normalize_observations", True)
    )
    episode_length = (
        config.episode_length
        if config.episode_length is not None
        else int(offline_config.get("episode_length", 1000))
    )
    if episode_length < 1:
        raise ValueError("episode_length must be positive.")

    return checkpoint_path, LoadedPolicy(
        env=env,
        task=task,
        dataset_name=dataset_name,
        actor_params=_actor_policy_params(checkpoint),
        actor_obs_stats=actor_obs_stats,
        decoder=decoder,
        normalize_actor_observations=normalize_actor_observations,
        episode_length=episode_length,
    )


def _policy_action(
    policy: LoadedPolicy,
    observation: jax.Array,
    key: jax.Array,
    deterministic: bool,
    clip_actions: bool,
) -> jax.Array:
    actor_observation = observation
    if policy.normalize_actor_observations:
        actor_observation = (
            observation - policy.actor_obs_stats.mean
        ) / (policy.actor_obs_stats.std + 1e-8)
    distribution = networks.gaussian_policy_fwd(
        policy.actor_params, actor_observation
    )
    latent = distribution.loc if deterministic else distribution.sample(key)
    action = policy.decoder.sample_action_from_z(
        observation, latent, key, deterministic=True
    )
    return jnp.clip(action, -1.0, 1.0) if clip_actions else action


def _evaluate_rollouts(
    policy: LoadedPolicy,
    episodes: int,
    seed: int,
    deterministic: bool,
    clip_actions: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if episodes < 1:
        raise ValueError("episodes must be positive.")

    reset_keys = jax.random.split(jax.random.key(seed), episodes)
    states = jax.vmap(policy.env.reset)(reset_keys)
    returns = jnp.zeros((episodes,), dtype=jnp.float32)
    lengths = jnp.zeros((episodes,), dtype=jnp.int32)
    active = jnp.ones((episodes,), dtype=bool)

    @jax.jit
    def step(
        states: Any,
        returns: jax.Array,
        lengths: jax.Array,
        active: jax.Array,
        key: jax.Array,
    ) -> tuple[Any, jax.Array, jax.Array, jax.Array]:
        action_keys = jax.random.split(key, episodes)
        actions = jax.vmap(
            lambda obs, action_key: _policy_action(
                policy, obs, action_key, deterministic, clip_actions
            )
        )(states.obs, action_keys)
        stepped_states = jax.vmap(policy.env.step)(states, actions)
        returns = returns + jnp.where(active, stepped_states.reward, 0.0)
        lengths = lengths + active.astype(jnp.int32)
        next_active = active & ~stepped_states.done.astype(bool)

        def retain_old(old: jax.Array, new: jax.Array) -> jax.Array:
            mask = next_active.reshape(
                next_active.shape + (1,) * (new.ndim - next_active.ndim)
            )
            return jnp.where(mask, new, old)

        states = jax.tree.map(retain_old, states, stepped_states)
        return states, returns, lengths, next_active

    rollout_key = jax.random.key(seed + 10_000)
    for _ in range(policy.episode_length):
        rollout_key, step_key = jax.random.split(rollout_key)
        states, returns, lengths, active = step(
            states, returns, lengths, active, step_key
        )
        if not bool(np.asarray(jnp.any(active))):
            break
    return np.asarray(returns), np.asarray(lengths)


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


def main(config: EvaluationConfig) -> None:
    checkpoint_path, policy = load_policy(config)
    returns, lengths = _evaluate_rollouts(
        policy,
        episodes=config.episodes,
        seed=config.seed,
        deterministic=config.deterministic,
        clip_actions=config.clip_actions,
    )
    random_score, expert_score = D4RL_SCORE_RANGES[policy.task]
    normalized_scores = (
        100.0 * (returns - random_score) / (expert_score - random_score)
    )
    result = {
        "checkpoint": str(checkpoint_path),
        "environment": policy.task,
        "d4rl_dataset": policy.dataset_name,
        "episodes": config.episodes,
        "episode_length_limit": policy.episode_length,
        "deterministic": config.deterministic,
        "clip_actions": config.clip_actions,
        "return": _summary(returns),
        "d4rl_normalized_score": _summary(normalized_scores),
        "episode_length": _summary(lengths.astype(np.float32)),
        "episode_returns": returns.tolist(),
        "episode_normalized_scores": normalized_scores.tolist(),
        "episode_lengths": lengths.tolist(),
    }
    print(json.dumps(result, indent=2))

    if config.output_json is not None:
        output_path = Path(config.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as file:
            json.dump(result, file, indent=2)
        print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main(tyro.cli(EvaluationConfig))
