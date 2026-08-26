"""Evaluate a frozen offline ResidualMLP policy on the local D4RL MJX envs.

This script loads checkpoints produced by
``run_offline_1step_fm_frozen_new_residualMLP.py``.  The checkpoint contains an
IQL-trained Gaussian latent policy (``ppo_z_params.policy``) and a frozen
one-step ResidualMLP flow decoder.  Evaluation uses the same inference path as
training:

    observation -> deterministic/stochastic latent z -> decoder -> clipped action

Example:

    python evaluate_offline_fm_residualMLP.py \
      --checkpoint results/.../checkpoint_final.pkl \
      --num-episodes 100 --deterministic
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

from d4rl_envs.mjx_envs import make_d4rl_env
from flow_policy import networks
from flow_policy.decoder_1step_fm_residualMLP import Decoder1StepFMState


# D4RL v2 reference returns used by get_normalized_score().
D4RL_SCORE_RANGES: dict[str, tuple[float, float]] = {
    "halfcheetah": (-280.178953, 12135.0),
    "hopper": (-20.272305, 3234.3),
    "walker2d": (1.629008, 4592.3),
    "ant": (-325.6, 3879.7),
}


@dataclass
class Config:
    """Command-line configuration for offline checkpoint evaluation."""

    checkpoint: str
    d4rl_dataset: str | None = "walker2d-medium-expert-v2"
    env_name: str | None = None
    num_episodes: int = 50
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
    decoder: Decoder1StepFMState
    normalize_actor_observations: bool
    episode_length: int


def _checkpoint_dict(path: str) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    with checkpoint_path.open("rb") as file:
        checkpoint = pickle.load(file)
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected a dictionary checkpoint, got {type(checkpoint).__name__}."
        )
    return checkpoint


def _offline_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = checkpoint.get("offline_config", {})
    if isinstance(config, dict):
        return config
    try:
        return asdict(config)
    except (TypeError, ValueError):
        return vars(config) if hasattr(config, "__dict__") else {}


def _resolve_environment(
    checkpoint: dict[str, Any], config: Config
) -> tuple[str, str]:
    offline_config = _offline_config(checkpoint)
    source = (
        config.env_name
        or config.d4rl_dataset
        or checkpoint.get("d4rl_dataset")
        or offline_config.get("d4rl_dataset")
        or checkpoint.get("env_name")
        or offline_config.get("env_name")
    )
    if source is None:
        raise ValueError(
            "Could not infer the environment. Pass --d4rl-dataset or --env-name."
        )

    dataset_name = str(source)
    family = dataset_name.lower().split("-", 1)[0]
    aliases = {
        "halfcheetah": "halfcheetah",
        "half_cheetah": "halfcheetah",
        "hopper": "hopper",
        "walker2d": "walker2d",
        "walker": "walker2d",
        "ant": "ant",
    }
    if family not in aliases:
        compact = family.replace("_", "").replace(" ", "")
        if compact == "halfcheetah":
            family = "halfcheetah"
        else:
            raise ValueError(
                f"Unsupported D4RL environment {source!r}. Supported families: "
                "ant, halfcheetah, hopper, walker2d."
            )
    return aliases[family], dataset_name


def _actor_policy_params(checkpoint: dict[str, Any]) -> Any:
    params = checkpoint.get("ppo_z_params")
    if params is None:
        raise KeyError(
            "Checkpoint is missing 'ppo_z_params'; use a checkpoint emitted by "
            "run_offline_1step_fm_frozen_new_residualMLP.py."
        )
    if hasattr(params, "policy"):
        return params.policy
    if isinstance(params, dict) and "policy" in params:
        return params["policy"]
    raise TypeError("'ppo_z_params' does not contain policy parameters.")


def _decoder_checkpoint_fields(
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
        raise KeyError("Checkpoint is missing decoder fields: " + ", ".join(missing))
    return params, obs_stats, decoder_config, int(obs_dim), int(action_dim)


def load_policy(config: Config) -> LoadedPolicy:
    checkpoint = _checkpoint_dict(config.checkpoint)
    task, dataset_name = _resolve_environment(checkpoint, config)
    env = make_d4rl_env(task)
    (
        decoder_params,
        decoder_obs_stats,
        decoder_config,
        obs_dim,
        action_dim,
    ) = _decoder_checkpoint_fields(checkpoint)

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

    decoder = Decoder1StepFMState.init(
        jax.random.key(config.seed + 1), obs_dim, action_dim, decoder_config
    )
    with jdc.copy_and_mutate(decoder) as decoder:
        decoder.params = decoder_params
        decoder.obs_stats = decoder_obs_stats

    offline_config = _offline_config(checkpoint)
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

    return LoadedPolicy(
        env=env,
        task=task,
        dataset_name=dataset_name,
        actor_params=_actor_policy_params(checkpoint),
        decoder=decoder,
        normalize_actor_observations=normalize_actor_observations,
        episode_length=episode_length,
    )


def _normalized_observation(policy: LoadedPolicy, observation: jax.Array) -> jax.Array:
    if not policy.normalize_actor_observations:
        return observation
    # The residual-MLP trainer shares the decoder's limits normalizer with IQL.
    return policy.decoder._normalize_obs(observation)


def _policy_action(
    policy: LoadedPolicy,
    observation: jax.Array,
    key: jax.Array,
    deterministic: bool,
    clip_actions: bool,
) -> jax.Array:
    normalized_observation = _normalized_observation(policy, observation)
    latent_distribution = networks.gaussian_policy_fwd(
        policy.actor_params, normalized_observation
    )
    latent = (
        latent_distribution.loc
        if deterministic
        else latent_distribution.sample(key)
    )
    action = policy.decoder.sample_action_from_z(
        observation,
        latent,
        key,
        deterministic=True,
    )
    return jnp.clip(action, -1.0, 1.0) if clip_actions else action


def _evaluate_rollouts(
    policy: LoadedPolicy,
    num_episodes: int,
    seed: int,
    deterministic: bool,
    clip_actions: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if num_episodes < 1:
        raise ValueError("num_episodes must be positive.")

    reset_keys = jax.random.split(jax.random.key(seed), num_episodes)
    states = jax.vmap(policy.env.reset)(reset_keys)
    returns = jnp.zeros((num_episodes,), dtype=jnp.float32)
    lengths = jnp.zeros((num_episodes,), dtype=jnp.int32)
    active = jnp.ones((num_episodes,), dtype=bool)

    @jax.jit
    def step(
        states: Any,
        returns: jax.Array,
        lengths: jax.Array,
        active: jax.Array,
        key: jax.Array,
    ) -> tuple[Any, jax.Array, jax.Array, jax.Array]:
        action_keys = jax.random.split(key, num_episodes)
        actions = jax.vmap(
            lambda obs, action_key: _policy_action(
                policy, obs, action_key, deterministic, clip_actions
            )
        )(states.obs, action_keys)
        stepped_states = jax.vmap(policy.env.step)(states, actions)
        returns = returns + jnp.where(active, stepped_states.reward, 0.0)
        lengths = lengths + active.astype(jnp.int32)
        next_active = jnp.logical_and(
            active, jnp.logical_not(stepped_states.done.astype(bool))
        )

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


def _normalized_scores(task: str, returns: np.ndarray) -> np.ndarray:
    random_score, expert_score = D4RL_SCORE_RANGES[task]
    return 100.0 * (returns - random_score) / (expert_score - random_score)


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


def main(config: Config) -> None:
    policy = load_policy(config)
    returns, lengths = _evaluate_rollouts(
        policy,
        num_episodes=config.num_episodes,
        seed=config.seed,
        deterministic=config.deterministic,
        clip_actions=config.clip_actions,
    )
    normalized_scores = _normalized_scores(policy.task, returns)
    result = {
        "checkpoint": str(Path(config.checkpoint).expanduser()),
        "environment": policy.task,
        "d4rl_dataset": policy.dataset_name,
        "num_episodes": config.num_episodes,
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
        output_path = Path(config.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as file:
            json.dump(result, file, indent=2)
        print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))