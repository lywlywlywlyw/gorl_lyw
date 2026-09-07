from ..base_env import BaseEnv, ObservationSize, State
from copy import deepcopy

# import utils.file_utils as FileUtils
# import utils.obs_utils as ObsUtils
# import utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robosuite
import h5py
from jax import numpy as jnp
import jax
import numpy as np


def _robosuite_supports_composite_controllers() -> bool:
    """Return whether the installed robosuite accepts v1.5 metadata."""
    try:
        major, minor = (int(part) for part in robosuite.__version__.split(".")[:2])
    except (AttributeError, TypeError, ValueError):
        # Do not rewrite metadata for an unknown/new development version.
        return True
    return (major, minor) >= (1, 5)


def _compatible_env_metadata(env_meta: dict) -> dict:
    """Translate robosuite 1.5 dataset metadata for robosuite 1.4."""
    env_meta = deepcopy(env_meta)
    if _robosuite_supports_composite_controllers():
        return env_meta

    env_kwargs = env_meta.setdefault("env_kwargs", {})
    # Added to the robosuite environment constructor in v1.5.
    env_kwargs.pop("lite_physics", None)

    controller = env_kwargs.get("controller_configs")
    if not isinstance(controller, dict) or controller.get("type") != "BASIC":
        return env_meta

    # v1.5 wraps a single-arm controller in a BASIC composite controller.
    # v1.4 expects the underlying arm controller dictionary directly.
    body_parts = controller.get("body_parts", {})
    arm_controllers = [
        config
        for config in body_parts.values()
        if isinstance(config, dict) and config.get("type") != "GRIP"
    ]
    if len(arm_controllers) != 1:
        raise ValueError(
            "This dataset uses a robosuite 1.5 composite controller that cannot "
            f"be represented by installed robosuite {robosuite.__version__}. "
            "Install robosuite >= 1.5 to use it."
        )
    controller = deepcopy(arm_controllers[0])
    controller.pop("input_ref_frame", None)
    controller.pop("gripper", None)
    env_kwargs["controller_configs"] = controller
    return env_meta


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
        env_meta = _compatible_env_metadata(
            FileUtils.get_env_metadata_from_dataset(self.dataset_path)
        )
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

    def _resolve_observation_sources(self, obs_dict: dict) -> dict[str, str]:
        """Map dataset observation keys to keys exposed by the live environment."""
        available_keys = set(obs_dict)
        source_keys = {}
        missing_keys = []
        for dataset_key in self.obs_keys:
            if dataset_key in available_keys:
                source_keys[dataset_key] = dataset_key
                continue

            # robosuite versions differ on whether site observables include the
            # ``_site`` suffix. Resolve this from the keys actually returned by
            # the current environment instead of assuming one fixed version.
            candidates = []
            if dataset_key.endswith("_site"):
                candidates.append(dataset_key[:-5])
            source_key = next(
                (candidate for candidate in candidates if candidate in available_keys),
                None,
            )
            if source_key is None:
                missing_keys.append(dataset_key)
            else:
                source_keys[dataset_key] = source_key

        if missing_keys:
            raise KeyError(
                "Robomimic environment observation is missing dataset keys "
                f"{missing_keys}; available keys: {sorted(obs_dict)}"
            )
        return source_keys

    def flatten_obs_dict(self, obs_dict: dict) -> jax.Array:
        source_keys = self._resolve_observation_sources(obs_dict)
        return jnp.concatenate(
            [jnp.ravel(jnp.asarray(obs_dict[source_keys[key]])) for key in self.obs_keys],
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

    def get_env_state(self) -> dict[str, np.ndarray | float | int | bool]:
        """Return the simulator and robosuite episode state for restoration."""
        sim = getattr(self.env, "sim", None)
        if sim is None:
            sim = getattr(getattr(self.env, "env", None), "sim", None)
        if sim is None:
            raise RuntimeError("The robomimic environment does not expose a MuJoCo sim.")
        data = sim.data
        result: dict[str, np.ndarray | float | int | bool] = {
            "qpos": np.array(data.qpos, copy=True),
            "qvel": np.array(data.qvel, copy=True),
            "time": float(data.time),
        }
        for name in ("act", "qacc_warmstart", "userdata"):
            value = getattr(data, name, None)
            if value is not None:
                result[name] = np.array(value, copy=True)

        robosuite_env = getattr(self.env, "env", self.env)
        if hasattr(robosuite_env, "timestep"):
            result["robosuite_timestep"] = int(robosuite_env.timestep)
        if hasattr(robosuite_env, "cur_time"):
            result["robosuite_cur_time"] = float(robosuite_env.cur_time)
        if hasattr(robosuite_env, "done"):
            result["robosuite_done"] = bool(robosuite_env.done)
        return result

    def set_env_state(
        self, state: dict[str, np.ndarray | float | int | bool]
    ) -> None:
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

        robosuite_env = getattr(self.env, "env", self.env)
        timestep = state.get("robosuite_timestep", state.get("episode_step"))
        if timestep is not None and hasattr(robosuite_env, "timestep"):
            robosuite_env.timestep = int(timestep)
        if "robosuite_cur_time" in state and hasattr(robosuite_env, "cur_time"):
            robosuite_env.cur_time = float(state["robosuite_cur_time"])
        elif timestep is not None and hasattr(robosuite_env, "cur_time"):
            control_timestep = getattr(robosuite_env, "control_timestep", None)
            if control_timestep is not None:
                robosuite_env.cur_time = int(timestep) * float(control_timestep)
        if "robosuite_done" in state and hasattr(robosuite_env, "done"):
            robosuite_env.done = bool(state["robosuite_done"])
        elif timestep is not None and hasattr(robosuite_env, "done"):
            horizon = getattr(robosuite_env, "horizon", None)
            ignore_done = bool(getattr(robosuite_env, "ignore_done", False))
            if horizon is not None:
                robosuite_env.done = int(timestep) >= int(horizon) and not ignore_done

    def reset_to_dataset_state(
        self,
        states: np.ndarray,
        episode_step: int,
        model: str | None = None,
        ep_meta: str | None = None,
    ) -> State:
        """Restore a Robomimic HDF5 transition state and return its observation."""
        payload: dict[str, object] = {"states": np.asarray(states)}
        if model is not None:
            payload["model"] = model
        if ep_meta is not None:
            payload["ep_meta"] = ep_meta
        # Reset controller goals and episode bookkeeping before replacing the
        # simulator coordinates. State-only reset_to does not do this itself.
        self.env.reset()
        obs_dict = self.env.reset_to(payload)
        robosuite_env = getattr(self.env, "env", self.env)
        if hasattr(robosuite_env, "timestep"):
            robosuite_env.timestep = int(episode_step)
        if hasattr(robosuite_env, "cur_time"):
            control_timestep = getattr(robosuite_env, "control_timestep", None)
            if control_timestep is not None:
                robosuite_env.cur_time = int(episode_step) * float(control_timestep)
        if hasattr(robosuite_env, "done"):
            robosuite_env.done = False
        return State(
            obs=self.flatten_obs_dict(obs_dict),
            reward=jnp.asarray(0.0),
            done=jnp.asarray(False),
            info={"success": self.is_success()},
        )

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
