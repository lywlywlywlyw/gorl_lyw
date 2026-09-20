"""A small ``BaseEnv`` adapter for the D4RL MuJoCo tasks.

The adapter intentionally keeps the D4RL dataset interface separate from the
rollout interface.  ``dataset_path`` may be a D4RL environment id (the usual
case) or a ``.npz``/``.pkl`` q-learning dataset.  The latter is useful when a
dataset has already been downloaded locally.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import jax
import numpy as np
import h5py
from jax import numpy as jnp

from envs.base_env import BaseEnv, ObservationSize, State


_TASK_IDS = {
    "halfcheetah": "HalfCheetah-v4",
    "hopper": "Hopper-v4",
    "walker2d": "Walker2d-v4",
}

# Official D4RL locomotion reference scores. These are not estimated from a
# particular dataset and are used only for reporting normalized returns.
D4RL_SCORE_RANGES = {
    "halfcheetah": (-280.178953, 12135.0),
    "hopper": (-20.272305, 3234.3),
    "walker2d": (1.629008, 4592.3),
}

def normalized_d4rl_return(mean_return: float, env_name: str) -> float:
    """Return the official D4RL normalized score in percent."""
    random_score, expert_score = D4RL_SCORE_RANGES[_base_task_name(env_name)]
    return 100.0 * (float(mean_return) - random_score) / (expert_score - random_score)


def infer_env_name(dataset_path: str) -> str:
    """Infer the registered D4RL task id from dataset metadata or its path."""
    path = Path(str(dataset_path)).expanduser()
    if path.is_file() and path.suffix in {".h5", ".hdf5"}:
        try:
            import h5py
            with h5py.File(path, "r") as handle:
                for key in ("env_name", "environment", "env"):
                    value = handle.attrs.get(key)
                    if value is not None:
                        if isinstance(value, bytes):
                            value = value.decode()
                        break
                else:
                    value = str(path)
        except Exception:
            value = str(path)
    else:
        value = str(dataset_path)
    value = str(value).lower().replace("_", "-")
    task = _base_task_name(value)
    # Preserve the D4RL version/dataset suffix when it is present.
    for suffix in ("-v3", "-v4", "-v5", "-random-v2", "-medium-v2", "-medium-replay-v2", "-medium-expert-v2", "-expert-v2"):
        if suffix in value:
            return task + suffix
    return {"halfcheetah": "halfcheetah-medium-v2", "hopper": "hopper-medium-v2", "walker2d": "walker2d-medium-v2"}[task]


def _base_task_name(name: str) -> str:
    value = name.lower().replace("_", "-")
    for task in _TASK_IDS:
        if task in value:
            return task
    raise ValueError(
        f"Unsupported D4RL task {name!r}; use HalfCheetah, Hopper, or Walker2d."
    )


def _make_env(env_name: str):
    try:
        import gym
    except ImportError:
        import gymnasium as gym

    # D4RL registers ids such as halfcheetah-medium-v2.  If D4RL is not
    # installed, the native MuJoCo task remains usable for online rollouts.
    try:
        return gym.make(env_name)
    except Exception as original:
        fallback = _TASK_IDS[_base_task_name(env_name)]
        if fallback == env_name:
            raise
        try:
            return gym.make(fallback)
        except Exception:
            raise original


class D4RLEnv(BaseEnv):
    """D4RL/Gym environment with the same methods as ``RobomimicEnv``."""

    def __init__(
        self,
        dataset_path: str | None = None,
        env_name: str | None = None,
        render_offscreen: bool = False,
        reward_shaping: bool = False,
    ):
        # For D4RL, dataset_path is commonly the registered environment id.
        self.dataset_path = dataset_path or env_name or "halfcheetah-medium-v2"
        self.env_name = env_name or infer_env_name(self.dataset_path)
        self.render_offscreen = render_offscreen
        self.reward_shaping = bool(reward_shaping)
        self.env = _make_env(self.env_name)
        self._closed = False
        self.obs_keys: list[str] = []
        self.shape_meta = {"ac_dim": int(np.prod(self.env.action_space.shape))}

    @property
    def action_size(self) -> int:
        return int(np.prod(self.env.action_space.shape))

    @property
    def observation_size(self) -> ObservationSize:
        return int(np.prod(self.env.observation_space.shape))

    @staticmethod
    def _observation(value: Any) -> jax.Array:
        return jnp.asarray(np.asarray(value, dtype=np.float32).reshape(-1))

    def reset(self, rng: jax.Array) -> State:
        seed = int(np.asarray(jax.random.key_data(rng))[0])
        try:
            result = self.env.reset(seed=seed)
        except TypeError:
            try:
                self.env.seed(seed)
            except AttributeError:
                pass
            result = self.env.reset()
        obs, info = (result if isinstance(result, tuple) else (result, {}))
        return State(self._observation(obs), jnp.asarray(0.0), jnp.asarray(False),
                     {**(info if isinstance(info, dict) else {}), "success": False})

    def step(self, state: State, action: jax.Array) -> State:
        action_np = np.asarray(action, dtype=np.float32).reshape(self.env.action_space.shape)
        if hasattr(self.env.action_space, "low"):
            action_np = np.clip(action_np, self.env.action_space.low, self.env.action_space.high)
        result = self.env.step(action_np)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            terminated = bool(terminated)
            truncated = bool(truncated)
            done = terminated or truncated
        else:
            obs, reward, done, info = result
            terminated = bool(done)
            truncated = False
        info = dict(info) if isinstance(info, dict) else {}
        # Keep true termination separate from time-limit truncation. The online
        # critic uses d4rl_terminal for its bootstrap discount.
        info["d4rl_terminal"] = terminated
        info["d4rl_timeout"] = truncated
        info.setdefault("success", False)
        return State(self._observation(obs), jnp.asarray(float(reward)),
                     jnp.asarray(done), info)

    def get_env_state(self) -> dict[str, Any]:
        """Return a best-effort MuJoCo state for rollout bookkeeping."""
        unwrapped = getattr(self.env, "unwrapped", self.env)
        data = getattr(unwrapped, "data", None)
        if data is not None and hasattr(data, "qpos") and hasattr(data, "qvel"):
            return {"qpos": np.asarray(data.qpos).copy(), "qvel": np.asarray(data.qvel).copy(),
                    "time": float(getattr(data, "time", 0.0))}
        return {"observation": np.asarray(getattr(self.env, "state", []), dtype=np.float32).copy()}

    def set_env_state(self, state: dict[str, Any]) -> None:
        unwrapped = getattr(self.env, "unwrapped", self.env)
        data = getattr(unwrapped, "data", None)
        if data is not None and "qpos" in state and "qvel" in state:
            data.qpos[:] = state["qpos"]
            data.qvel[:] = state["qvel"]
            if "time" in state and hasattr(data, "time"):
                data.time = state["time"]
            if hasattr(unwrapped, "set_state"):
                unwrapped.set_state(np.asarray(data.qpos), np.asarray(data.qvel))

    def reset_to_dataset_state(self, states=None, episode_step: int = 0, **kwargs) -> State:
        if isinstance(states, dict):
            self.set_env_state(states)
            obs = self.env._get_obs() if hasattr(self.env, "_get_obs") else self.env.reset()[0]
        else:
            result = self.env.reset()
            obs = result[0] if isinstance(result, tuple) else result
        return State(self._observation(obs), jnp.asarray(0.0), jnp.asarray(False), {"success": False})

    def get_normalized_score(self, mean_return: float) -> float:
        """Return the official D4RL normalized score in [roughly] percent."""
        task = _base_task_name(self.env_name)
        random_score, expert_score = D4RL_SCORE_RANGES[task]
        return (float(mean_return) - random_score) / (expert_score - random_score)

    def get_dataset(self) -> dict[str, np.ndarray]:
        path = Path(str(self.dataset_path)).expanduser()
        if path.is_file():
            if path.suffix in {".h5", ".hdf5"}:
                with h5py.File(path, "r") as handle:
                    required = ("observations", "actions", "rewards", "next_observations", "dones")
                    missing = [key for key in required if key not in handle]
                    if missing:
                        raise KeyError(
                            f"Processed D4RL dataset is missing {missing}; run the preprocessing script first."
                        )
                    raw = {key: np.asarray(handle[key]) for key in handle.keys()}
            elif path.suffix == ".npz":
                raw = dict(np.load(path, allow_pickle=False))
            elif path.suffix in {".pkl", ".pickle"}:
                with path.open("rb") as file:
                    raw = pickle.load(file)
            else:
                raise ValueError(f"Unsupported D4RL dataset file: {path}")
        else:
            raw = None
            try:
                import d4rl  # noqa: F401
                import d4rl as _d4rl
                raw = _d4rl.qlearning_dataset(self.env)
            except Exception:
                if hasattr(self.env, "get_dataset"):
                    raw = self.env.get_dataset()
            if raw is None:
                raise RuntimeError("D4RL is not installed and the environment has no get_dataset().")
        result = {key: np.asarray(value) for key, value in raw.items()}
        if "dones" not in result:
            raise KeyError("D4RL training data must contain dones; preprocess the source dataset first.")
        if "next_observations" not in result:
            result["next_observations"] = np.concatenate([result["observations"][1:], result["observations"][-1:]])
        return result

    def render(self, mode="human", height=None, width=None, camera_name=None):
        try:
            return self.env.render()
        except TypeError:
            return self.env.render(mode=mode)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            close = getattr(self.env, "close", None)
            if callable(close):
                close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
