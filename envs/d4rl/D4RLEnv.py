"""Minari MuJoCo dataset adapter backed by Gymnasium v5 environments.

``dataset_path`` is a Minari dataset id such as
``mujoco/walker2d/medium-v0``. Legacy local files remain accepted.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import jax
import numpy as np
import h5py
from jax import numpy as jnp

from envs.base_env import BaseEnv, ObservationSize, State


_TASK_IDS = {
    "halfcheetah": "HalfCheetah-v5",
    "hopper": "Hopper-v5",
    "walker2d": "Walker2d-v5",
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
    return _TASK_IDS[task]


def _base_task_name(name: str) -> str:
    value = name.lower().replace("_", "-")
    for task in _TASK_IDS:
        if task in value:
            return task
    raise ValueError(
        f"Unsupported D4RL task {name!r}; use HalfCheetah, Hopper, or Walker2d."
    )


def _make_env(env_name: str, render_mode: str | None = None):
    try:
        import gym
    except ImportError:
        import gymnasium as gym

    make_kwargs = {} if render_mode is None else {"render_mode": render_mode}
    return gym.make(env_name, **make_kwargs)


class D4RLEnv(BaseEnv):
    """D4RL/Gym environment with the same methods as ``RobomimicEnv``."""

    def __init__(
        self,
        dataset_path: str | None = None,
        env_name: str | None = None,
        render_offscreen: bool = False,
        reward_shaping: bool = False,
    ):
        # Minari dataset IDs select the corresponding v5 task by name.
        self.dataset_path = dataset_path or env_name or "mujoco/walker2d/medium-v0"
        self.env_name = env_name or infer_env_name(self.dataset_path)
        self.render_offscreen = render_offscreen
        self.reward_shaping = bool(reward_shaping)
        # Gymnasium selects the renderer when the environment is created.
        # Video evaluation requests RGB frames, so configure that mode here;
        # calling render(mode=...) later cannot change an already-created
        # MuJoCo renderer.
        self.env = _make_env(
            self.env_name,
            render_mode="rgb_array" if render_offscreen else None,
        )
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

    def _processed_cache_paths(self) -> tuple[Path, Path]:
        """Return the canonical processed HDF5 and PKL paths for a Minari id."""
        cache_dir = Path("/root/GoRL/datasets/d4rl")
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = str(self.dataset_path).strip("/").replace("/", "-").replace("_", "-")
        return cache_dir / f"{stem}_processed.hdf5", cache_dir / f"{stem}_processed.pkl"

    @staticmethod
    def _write_processed_cache(
        data: dict[str, np.ndarray], hdf5_path: Path, pkl_path: Path, dataset_id: str
    ) -> None:
        """Atomically store identical processed arrays in HDF5 and PKL formats."""
        hdf5_tmp = hdf5_path.with_suffix(hdf5_path.suffix + ".tmp")
        pkl_tmp = pkl_path.with_suffix(pkl_path.suffix + ".tmp")
        hdf5_tmp.unlink(missing_ok=True)
        pkl_tmp.unlink(missing_ok=True)
        try:
            with h5py.File(hdf5_tmp, "w") as handle:
                handle.attrs["dataset_id"] = dataset_id
                handle.attrs["env_name"] = infer_env_name(dataset_id)
                for key, value in data.items():
                    handle.create_dataset(key, data=np.asarray(value))
                handle.flush()
            with pkl_tmp.open("wb") as file:
                pickle.dump({key: np.asarray(value) for key, value in data.items()}, file, protocol=pickle.HIGHEST_PROTOCOL)
                file.flush()
                os.fsync(file.fileno())
            os.replace(hdf5_tmp, hdf5_path)
            os.replace(pkl_tmp, pkl_path)
        except BaseException:
            hdf5_tmp.unlink(missing_ok=True)
            pkl_tmp.unlink(missing_ok=True)
            raise

    def get_dataset(self) -> dict[str, np.ndarray]:
        path = Path(str(self.dataset_path)).expanduser()
        cache_hdf5, cache_pkl = self._processed_cache_paths()
        if not path.is_file() and cache_hdf5.is_file():
            path = cache_hdf5
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
            try:
                import minari
                dataset = minari.load_dataset(str(self.dataset_path), download=True)
            except Exception as error:
                raise RuntimeError(f"Could not load Minari dataset {self.dataset_path!r}.") from error
            observations, actions, rewards, next_observations = [], [], [], []
            dones, truncations = [], []
            for episode in dataset.iterate_episodes():
                obs = np.asarray(episode.observations, dtype=np.float32)
                act = np.asarray(episode.actions, dtype=np.float32)
                rew = np.asarray(episode.rewards, dtype=np.float32).reshape(-1)
                term = np.asarray(episode.terminations, dtype=np.bool_).reshape(-1)
                trunc = np.asarray(episode.truncations, dtype=np.bool_).reshape(-1)
                count = len(act)
                if len(obs) != count + 1 or any(len(x) != count for x in (rew, term, trunc)):
                    raise ValueError(f"Malformed Minari episode in {self.dataset_path!r}.")
                observations.append(obs[:-1]); next_observations.append(obs[1:])
                actions.append(act); rewards.append(rew); dones.append(term); truncations.append(trunc)
            if not observations:
                raise RuntimeError(f"Minari dataset {self.dataset_path!r} contains no episodes.")
            raw = {"observations": np.concatenate(observations), "next_observations": np.concatenate(next_observations),
                   "actions": np.concatenate(actions), "rewards": np.concatenate(rewards),
                   "dones": np.concatenate(dones), "truncations": np.concatenate(truncations)}
            raw["episode_ends"] = np.logical_or(raw["dones"], raw["truncations"])
            self._write_processed_cache(raw, cache_hdf5, cache_pkl, str(self.dataset_path))
            path = cache_hdf5
        result = {key: np.asarray(value) for key, value in raw.items()}
        # Keep the PKL cache synchronized even if an older run created only HDF5.
        if path == cache_hdf5 and not cache_pkl.exists():
            with cache_pkl.open("wb") as file:
                pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
                file.flush()
                os.fsync(file.fileno())
        if "dones" not in result:
            raise KeyError("Dataset must contain Minari terminations/dones.")
        if "next_observations" not in result:
            result["next_observations"] = np.concatenate([result["observations"][1:], result["observations"][-1:]])
        return result

    def render(self, mode="human", height=None, width=None, camera_name=None):
        # Gymnasium fixes the render mode at environment creation time.  The
        # offscreen environment is created with rgb_array, so render directly
        # and resize to the dimensions requested by the video writer.
        try:
            frame = self.env.render()
        except TypeError:
            frame = self.env.render(mode=mode)
        if frame is not None and width is not None and height is not None:
            frame = np.asarray(frame)
            if frame.ndim >= 2 and (frame.shape[1] != int(width) or frame.shape[0] != int(height)):
                import cv2
                frame = cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_AREA)
        return frame

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
