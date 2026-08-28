from ..base_env import BaseEnv, ObservationSize, State
# import utils.file_utils as FileUtils
# import utils.obs_utils as ObsUtils
# import utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import h5py
from jax import numpy as jnp
import jax
import numpy as np
class RobomimicEnv(BaseEnv):
    def __init__(
        self,
        dataset_path: str,
        render_offscreen: bool = False,
        reward_shaping: bool = True,
    ):
        # Keep construction / cleanup safe when this class is repeatedly created
        # inside spawned rollout workers. In particular, robosuite owns native
        # MuJoCo / EGL resources that must be released explicitly instead of being
        # left to Python interpreter shutdown.
        self.env = None
        self._closed = False
        self.dataset_path = dataset_path
        self.render_offscreen = render_offscreen
        self.reward_shaping = reward_shaping
        try:
            self.obs_keys = self.load_dataset()
            self.env = self.load_env()
            # Environment construction imports several robomimic modules. Rebuild the
            # process-global modality map immediately before shape inference instead
            # of relying on initialization side effects from load_env().
            self.initialize_obs_modalities()
            self.shape_meta = FileUtils.get_shape_metadata_from_dataset(
                dataset_config={"path": self.dataset_path},
                action_keys=["actions"],
                all_obs_keys=self.obs_keys,
                verbose=True,
            )
        except BaseException:
            try:
                self.close()
            except Exception:
                # Preserve the construction error; cleanup is best effort here.
                pass
            raise

    def initialize_obs_modalities(self):
        ObsUtils.initialize_obs_modality_mapping_from_dict(
            modality_mapping={
                "low_dim": self.obs_keys,
            }
        )

    def load_env(self):
        self.initialize_obs_modalities()
        env_meta = FileUtils.get_env_metadata_from_dataset(self.dataset_path)
        # Override the dataset setting before robomimic constructs robosuite.
        # This is forwarded by create_env_from_metadata -> robosuite.make.
        env_meta.setdefault("env_kwargs", {})["reward_shaping"] = self.reward_shaping
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=self.render_offscreen,
            use_image_obs=False,
            use_depth_obs=False,
        )
        return env

    def load_dataset(self):
        with h5py.File(self.dataset_path, "r") as f:
            demo_key = next(iter(f["data"].keys()))
            obs_keys = list(f[f"data/{demo_key}/obs"].keys())
        return obs_keys
        
    def reset(self, rng: jax.Array) -> State:
        obs_dict = self.env.reset()
        obs = self.flatten_obs_dict(obs_dict)
        return State(
            obs=obs,
            reward=jnp.array(0.0),
            done=jnp.array(False),
            info={},
        )

    def flatten_obs_dict(self, obs_dict: dict) -> jax.Array:
        missing_keys = [key for key in self.obs_keys if key not in obs_dict]
        if missing_keys:
            raise KeyError(
                "Robomimic environment observation is missing dataset keys "
                f"{missing_keys}; available keys: {sorted(obs_dict)}"
            )
        return jnp.concatenate(
            [jnp.ravel(jnp.asarray(obs_dict[key])) for key in self.obs_keys],
            axis=0,
        )

    def step(self, state: State, action: jax.Array) -> State:
        obs_dict, reward, done, info = self.env.step(np.asarray(action))
        obs = self.flatten_obs_dict(obs_dict)
        info = dict(info or {})
        info["success"] = self.is_success()
        return State(
            obs=obs,
            reward=jnp.asarray(reward),
            done=jnp.asarray(done),
            info=info,
        )

    def get_env_state(self) -> dict[str, np.ndarray | float]:
        """Return a copy of the MuJoCo simulator state for exact restoration."""
        sim = getattr(self.env, "sim", None)
        if sim is None:
            sim = getattr(getattr(self.env, "env", None), "sim", None)
        if sim is None:
            raise RuntimeError("The robomimic environment does not expose a MuJoCo sim.")
        data = sim.data
        result: dict[str, np.ndarray | float] = {
            "qpos": np.array(data.qpos, copy=True),
            "qvel": np.array(data.qvel, copy=True),
            "time": float(data.time),
        }
        for name in ("act", "qacc_warmstart", "userdata"):
            value = getattr(data, name, None)
            if value is not None:
                result[name] = np.array(value, copy=True)
        return result

    def set_env_state(self, state: dict[str, np.ndarray | float]) -> None:
        """Restore a state produced by :meth:`get_env_state`."""
        sim = getattr(self.env, "sim", None)
        if sim is None:
            sim = getattr(getattr(self.env, "env", None), "sim", None)
        if sim is None:
            raise RuntimeError("The robomimic environment does not expose a MuJoCo sim.")
        data = sim.data
        data.qpos[:] = state["qpos"]
        data.qvel[:] = state["qvel"]
        data.time = state["time"]
        for name in ("act", "qacc_warmstart", "userdata"):
            if name in state and hasattr(data, name):
                getattr(data, name)[:] = state[name]
        sim.forward()

    def is_success(self) -> bool:
        """Return the wrapped Robomimic task-level success signal."""
        success = self.env.is_success()
        if isinstance(success, dict):
            success = success.get("task", False)
        return bool(np.asarray(success))

    def close(self) -> None:
        """Idempotently release the wrapped robosuite MuJoCo / EGL resources."""
        if self._closed:
            return
        self._closed = True

        env = self.env
        self.env = None
        if env is None:
            return

        close = getattr(env, "close", None)
        if callable(close):
            close()
            return

        # robomimic's EnvRobosuite wrapper does not expose close(), but its
        # ``env`` member is the actual robosuite MujocoEnv and does. Calling
        # that public method releases MjSim and its offscreen EGL context before
        # the spawned worker interpreter starts tearing modules down.
        wrapped_env = getattr(env, "env", None)
        wrapped_close = getattr(wrapped_env, "close", None)
        if callable(wrapped_close):
            wrapped_close()

    def __del__(self) -> None:
        """Best-effort fallback for callers that forget to close the environment."""
        try:
            self.close()
        except Exception:
            # Destructors run during partially torn-down interpreter state and must
            # never mask the original worker error.
            pass
    
    @property
    def action_size(self) -> int:
        return self.shape_meta["ac_dim"]

    @property
    def observation_size(self) -> ObservationSize:
        return sum(v[0] for v in self.shape_meta["all_shapes"].values())

    def render(
        self, mode="human", height=None, width=None, camera_name=None
    ):
        return self.env.render(mode=mode, height=height, width=width, camera_name=camera_name)