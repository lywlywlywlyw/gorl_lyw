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
    def __init__(self, dataset_path: str, render_offscreen: bool = False):
        self.dataset_path = dataset_path
        self.render_offscreen = render_offscreen
        self.obs_keys = self.load_dataset()
        self.env = self.load_env()
        self.shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_config={"path": self.dataset_path},
            action_keys=["actions"],
            all_obs_keys=self.obs_keys,
            verbose=True,
        )

    def load_env(self):
        ObsUtils.initialize_obs_modality_mapping_from_dict(
            modality_mapping={
                "low_dim": self.obs_keys,
            }
        )
        env_meta = FileUtils.get_env_metadata_from_dataset(self.dataset_path)
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
        return jnp.concatenate(
            [jnp.ravel(jnp.asarray(v)) for v in obs_dict.values()],
            axis=0,
        )

    def step(self, state: State, action: jax.Array) -> State:
        obs_dict, reward, done, info = self.env.step(np.asarray(action))
        obs = self.flatten_obs_dict(obs_dict)
        return State(
            obs=obs,
            reward=jnp.asarray(reward),
            done=jnp.asarray(done),
            info=info,
        )
    
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