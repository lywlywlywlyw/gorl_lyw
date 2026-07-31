"""Online MuJoCo evaluation for frozen and alternating offline-FM checkpoints."""

from __future__ import annotations

import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


@dataclass
class EvaluationConfig:
    """Evaluate an offline-FM checkpoint by running episodes in a simulator."""

    checkpoint: str
    env_id: str | None = None
    episodes: int = 10
    seed: int = 0
    gui: bool = False
    deterministic: bool = True
    clip_actions: bool = True
    max_episode_steps: int | None = None
    render_delay: float = 0.0
    output_json: str | None = None


def _load_gym() -> Any:
    try:
        import gymnasium as gym

        return gym
    except ImportError:
        try:
            import gym

            return gym
        except ImportError as error:
            raise ImportError(
                "Online evaluation requires Gymnasium with MuJoCo support. "
                "Install it with `pip install \"gymnasium[mujoco]\"`."
            ) from error


def _environment_candidates(dataset_id: str | None) -> list[str]:
    if dataset_id is None:
        return []
    environment = dataset_id.split("-", 1)[0].lower()
    names = {
        "walker2d": "Walker2d",
        "halfcheetah": "HalfCheetah",
        "hopper": "Hopper",
        "ant": "Ant",
    }
    if environment not in names:
        return []
    # D4RL locomotion data predates Gymnasium's v5 physics/reward fixes. The v4
    # port is therefore a closer modern approximation; retain v5 as fallback.
    return [f"{names[environment]}-v{version}" for version in (4, 5, 3, 2)]


def _make_environment(
    gym: Any, env_id: str | None, dataset_id: str | None, gui: bool
) -> tuple[Any, str]:
    candidates = [env_id] if env_id is not None else _environment_candidates(dataset_id)
    if not candidates:
        raise ValueError(
            "Could not infer a simulator from the checkpoint. Pass --env-id, "
            "for example --env-id Walker2d-v5."
        )
    failures: list[str] = []
    for candidate in candidates:
        try:
            kwargs = {"render_mode": "human"} if gui else {}
            return gym.make(candidate, **kwargs), candidate
        except Exception as error:
            failures.append(f"{candidate}: {error}")
    raise RuntimeError(
        "Could not create a compatible environment:\n  " + "\n  ".join(failures)
    )


def _validate_checkpoint(checkpoint: Any, path: Path) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path} does not contain a checkpoint dictionary.")
    required = {
        "config",
        "encoder_params",
        "observation_mean",
        "observation_std",
        "decoder",
    }
    missing = required - checkpoint.keys()
    if missing:
        raise KeyError(f"{path} is missing checkpoint fields: {sorted(missing)}")
    decoder_required = {"params", "obs_stats", "config"}
    decoder_missing = decoder_required - checkpoint["decoder"].keys()
    if decoder_missing:
        raise KeyError(
            f"{path} is missing decoder fields: {sorted(decoder_missing)}"
        )
    return checkpoint


def _method_name(checkpoint: dict[str, Any], path: Path) -> str:
    output_dir = str(checkpoint.get("config", {}).get("output_dir", "")).lower()
    path_text = str(path).lower()
    if "alternating" in output_dir or "alternating" in path_text:
        return "alternating"
    if "frozen" in output_dir or "frozen" in path_text:
        return "frozen_decoder"
    return "offline_fm"


def _reset(env: Any, seed: int) -> np.ndarray:
    result = env.reset(seed=seed)
    observation = result[0] if isinstance(result, tuple) else result
    return np.asarray(observation, dtype=np.float32)


def _step(env: Any, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
    result = env.step(action)
    if len(result) == 5:
        observation, reward, terminated, truncated, _ = result
        done = bool(terminated or truncated)
    else:
        observation, reward, done, _ = result
        done = bool(done)
    return np.asarray(observation, dtype=np.float32), float(reward), done


def main(config: EvaluationConfig) -> None:
    if config.episodes < 1:
        raise ValueError("episodes must be positive.")
    if config.max_episode_steps is not None and config.max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive when provided.")
    if config.render_delay < 0:
        raise ValueError("render_delay cannot be negative.")

    # Importing these lazily lets --help work before the accelerator stack is
    # installed and gives a focused simulator dependency error.
    import jax
    import jax_dataclasses as jdc
    from jax import numpy as jnp

    from flow_policy import networks
    from flow_policy.decoder_fm import DecoderFMState

    checkpoint_path = Path(config.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    with checkpoint_path.open("rb") as file:
        checkpoint = _validate_checkpoint(pickle.load(file), checkpoint_path)

    actor_params = checkpoint["encoder_params"]
    obs_mean = jnp.asarray(checkpoint["observation_mean"])
    obs_std = jnp.asarray(checkpoint["observation_std"])
    decoder_data = checkpoint["decoder"]
    obs_dim = int(obs_mean.shape[-1])
    action_dim = int(actor_params[-1][1].shape[-1] // 2)
    decoder = DecoderFMState.init(
        jax.random.key(config.seed),
        obs_dim,
        action_dim,
        decoder_data["config"],
    )
    with jdc.copy_and_mutate(decoder) as decoder:
        decoder.params = decoder_data["params"]
        decoder.obs_stats = decoder_data["obs_stats"]

    @jax.jit
    def policy(observation: Any, key: Any) -> Any:
        observation = jnp.asarray(observation)
        normalized = (observation - obs_mean) / (obs_std + 1e-8)
        distribution = networks.gaussian_policy_fwd(actor_params, normalized)
        latent_key, decoder_key = jax.random.split(key)
        if config.deterministic:
            latent = distribution.loc
        else:
            latent = distribution.loc + distribution.scale * jax.random.normal(
                latent_key, distribution.loc.shape
            )
        return decoder.sample_action_from_z(
            observation, latent, decoder_key, deterministic=True
        )

    gym = _load_gym()
    dataset_id = checkpoint.get("config", {}).get("d4rl_dataset")
    env, resolved_env_id = _make_environment(
        gym, config.env_id, dataset_id, config.gui
    )
    method = _method_name(checkpoint, checkpoint_path)
    if tuple(env.observation_space.shape) != (obs_dim,):
        env.close()
        raise ValueError(
            f"Environment observation shape {env.observation_space.shape} does "
            f"not match checkpoint dimension {(obs_dim,)}."
        )
    if tuple(env.action_space.shape) != (action_dim,):
        env.close()
        raise ValueError(
            f"Environment action shape {env.action_space.shape} does not match "
            f"checkpoint dimension {(action_dim,)}."
        )

    print(
        f"Evaluating {method} checkpoint on {resolved_env_id} for "
        f"{config.episodes} episodes ({'GUI' if config.gui else 'headless'}, "
        f"{'deterministic' if config.deterministic else 'stochastic'} policy)."
    )
    key = jax.random.key(config.seed)
    returns: list[float] = []
    lengths: list[int] = []
    try:
        for episode in range(config.episodes):
            observation = _reset(env, config.seed + episode)
            episode_return = 0.0
            episode_length = 0
            done = False
            while not done:
                key, action_key = jax.random.split(key)
                action = np.asarray(policy(observation, action_key))
                if config.clip_actions:
                    action = np.clip(
                        action, env.action_space.low, env.action_space.high
                    )
                observation, reward, done = _step(env, action)
                episode_return += reward
                episode_length += 1
                if (
                    config.max_episode_steps is not None
                    and episode_length >= config.max_episode_steps
                ):
                    done = True
                if config.gui and config.render_delay:
                    time.sleep(config.render_delay)
            returns.append(episode_return)
            lengths.append(episode_length)
            print(
                f"Episode {episode + 1:>3}/{config.episodes}: "
                f"return={episode_return:.3f}, length={episode_length}"
            )
    finally:
        env.close()

    result = {
        "checkpoint": str(checkpoint_path),
        "method": method,
        "environment": resolved_env_id,
        "dataset": dataset_id,
        "episodes": config.episodes,
        "seed": config.seed,
        "deterministic": config.deterministic,
        "clip_actions": config.clip_actions,
        "returns": returns,
        "lengths": lengths,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "length_mean": float(np.mean(lengths)),
    }
    print(
        "\nReturn summary: "
        f"mean={result['return_mean']:.3f}, std={result['return_std']:.3f}, "
        f"min={result['return_min']:.3f}, max={result['return_max']:.3f}"
    )
    if config.output_json is not None:
        output_path = Path(config.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as file:
            json.dump(result, file, indent=2)
        print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main(tyro.cli(EvaluationConfig))
