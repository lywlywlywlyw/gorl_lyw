"""D4RL Gym-MuJoCo locomotion tasks implemented with Playground and MJX.

The D4RL v2 suffix identifies the dataset revision, not a Gym environment
revision. These environments port the task definitions used by the legacy Gym
MuJoCo locomotion datasets while running the physics with modern MJX. The exact legacy Gym XML assets are vendored alongside this module. Walker2d
and Hopper use coordinate-only conversions for MuJoCo 3/MJX because their
original XML uses the removed ``coordinate="global"`` compiler option.

MJX and the original mujoco_py runtime are different physics implementations,
so this is a high-fidelity compatibility port rather than bitwise replay.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env


@dataclass(frozen=True)
class _TaskSpec:
    xml_name: str
    runtime_xml_name: str
    xml_sha256: str
    runtime_xml_sha256: str
    ctrl_dt: float
    sim_dt: float
    reset_noise: float
    ctrl_cost_weight: float
    obs_size: int
    action_size: int


_SPECS: Mapping[str, _TaskSpec] = {
    "walker2d": _TaskSpec(
        "walker2d.xml", "walker2d_mjx.xml",
        "88dff0833ee2c6ec30854c7616b86d2ce5bd5b26ea0cd6367768c5aca4700296",
        "ac13373c7ac9003a5311c4704b314ed11149b3dd7600bbc02770487300f9ad2d",
        0.008, 0.002, 0.005, 0.001, 17, 6,
    ),
    "hopper": _TaskSpec(
        "hopper.xml", "hopper_mjx.xml",
        "a181c527c392c14b8e41de9e64d3a235db3279a0c3fb74eaec99170f5d87ec79",
        "3ce93a055ffdcd83c0c701d2400768e40d2cbb9532f3c4ae33377c27f8b39f9e",
        0.008, 0.002, 0.005, 0.001, 11, 3,
    ),
    "halfcheetah": _TaskSpec(
        "half_cheetah.xml", "half_cheetah.xml",
        "11797a5d69e8ac955e89ca6fdd3a0087f1c990094fda401ac51420de1b6c5494",
        "11797a5d69e8ac955e89ca6fdd3a0087f1c990094fda401ac51420de1b6c5494",
        0.05, 0.01, 0.1, 0.1, 17, 6,
    ),
    "ant": _TaskSpec(
        "ant.xml", "ant.xml",
        "cd5f83ef0ea35b0969e65d360c5bacd5b74ccaef6b27e4433b5168c605e3e2be",
        "cd5f83ef0ea35b0969e65d360c5bacd5b74ccaef6b27e4433b5168c605e3e2be",
        0.05, 0.01, 0.1, 0.5, 111, 8,
    ),
}

_ALIASES = {
    "walker": "walker2d",
    "walker2d": "walker2d",
    "hopper": "hopper",
    "halfcheetah": "halfcheetah",
    "half-cheetah": "halfcheetah",
    "ant": "ant",
}


_ASSET_DIR = Path(__file__).resolve().parent / "assets"


def _checked_asset_path(name: str, expected_sha256: str) -> Path:
    """Returns a vendored asset after checking its provenance hash."""
    path = _ASSET_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Vendored D4RL MuJoCo asset not found: {path}")
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"D4RL MuJoCo asset hash mismatch for {path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return path


def normalize_task(task: str) -> str:
    family = task.split("-", 1)[0].lower()
    if family not in _ALIASES:
        raise ValueError(
            f"Unsupported D4RL MJX task {task!r}; supported: {sorted(_SPECS)}"
        )
    return _ALIASES[family]


def default_config(task: str) -> config_dict.ConfigDict:
    family = normalize_task(task)
    spec = _SPECS[family]
    return config_dict.create(
        ctrl_dt=spec.ctrl_dt,
        sim_dt=spec.sim_dt,
        episode_length=1000,
        action_repeat=1,
        vision=False,
    )


def make_d4rl_env(task: str) -> "D4RLMujocoEnv":
    """Creates a D4RL-compatible MuJoCo Playground environment."""
    family = normalize_task(task)
    return D4RLMujocoEnv(family, default_config(family))


class D4RLMujocoEnv(mjx_env.MjxEnv):
    """Legacy D4RL locomotion task exposed as a Playground MjxEnv."""

    def __init__(
        self,
        task: str,
        config: config_dict.ConfigDict,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        self._task = normalize_task(task)
        self._spec = _SPECS[self._task]
        super().__init__(config, config_overrides)
        if self._config.vision:
            raise NotImplementedError("D4RL-compatible MJX environments are state-only.")

        # Verify the byte-exact D4RL source. Walker2d/Hopper need their
        # hash-locked coordinate conversion because MuJoCo 3 removed global
        # coordinates; Ant/HalfCheetah compile the original directly.
        _checked_asset_path(self._spec.xml_name, self._spec.xml_sha256)
        xml_path = _checked_asset_path(
            self._spec.runtime_xml_name, self._spec.runtime_xml_sha256
        )
        self._xml_path = str(xml_path)
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model)
        self._init_qpos = jnp.asarray(self._mj_model.qpos0)
        self._init_qvel = jnp.zeros(self._mj_model.nv)
        self._torso_id = (
            self._mj_model.body("torso").id if self._task == "ant" else -1
        )

        if self._mj_model.nu != self._spec.action_size:
            raise ValueError(
                f"{self._task} XML action size {self._mj_model.nu} != "
                f"expected {self._spec.action_size}."
            )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, qpos_rng, qvel_rng = jax.random.split(rng, 3)
        scale = self._spec.reset_noise
        qpos = self._init_qpos + jax.random.uniform(
            qpos_rng, (self._mj_model.nq,), minval=-scale, maxval=scale
        )
        if self._task in {"halfcheetah", "ant"}:
            qvel_noise = scale * jax.random.normal(
                qvel_rng, (self._mj_model.nv,)
            )
        else:
            qvel_noise = jax.random.uniform(
                qvel_rng,
                (self._mj_model.nv,),
                minval=-scale,
                maxval=scale,
            )
        data = mjx_env.init(
            self.mjx_model, qpos=qpos, qvel=self._init_qvel + qvel_noise
        )
        zero = jnp.zeros(())
        metrics = {
            "reward_forward": zero,
            "reward_ctrl": zero,
            "reward_healthy": zero,
            "reward_contact": zero,
            "x_position": zero,
            "x_velocity": zero,
        }
        return mjx_env.State(
            data=data,
            obs=self._get_obs(data),
            reward=zero,
            done=zero,
            metrics=metrics,
            info={"rng": rng},
        )

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jnp.clip(action, -1.0, 1.0)
        x_before = self._x_position(state.data)
        data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
        x_after = self._x_position(data)
        x_velocity = (x_after - x_before) / self.dt

        forward_reward = x_velocity
        ctrl_cost = self._spec.ctrl_cost_weight * jnp.sum(jnp.square(action))
        healthy = self._is_healthy(data)
        healthy_reward = (
            jnp.zeros(()) if self._task == "halfcheetah" else jnp.ones(())
        )
        contact_cost = self._contact_cost(data)
        reward = forward_reward + healthy_reward - ctrl_cost - contact_cost
        done = jnp.zeros(()) if self._task == "halfcheetah" else 1.0 - healthy
        finite = jnp.isfinite(data.qpos).all() & jnp.isfinite(data.qvel).all()
        done = jnp.where(finite, done, 1.0).astype(jnp.float32)

        metrics = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_healthy": healthy_reward,
            "reward_contact": -contact_cost,
            "x_position": x_after,
            "x_velocity": x_velocity,
        }
        return mjx_env.State(
            data=data,
            obs=self._get_obs(data),
            reward=reward,
            done=done,
            metrics=metrics,
            info=state.info,
        )

    def _x_position(self, data: mjx.Data) -> jax.Array:
        if self._task == "ant":
            return data.xpos[self._torso_id, 0]
        return data.qpos[0]

    def _is_healthy(self, data: mjx.Data) -> jax.Array:
        finite = jnp.isfinite(data.qpos).all() & jnp.isfinite(data.qvel).all()
        if self._task == "halfcheetah":
            return finite.astype(jnp.float32)
        if self._task == "ant":
            z = data.qpos[2]
            return (finite & (z >= 0.2) & (z <= 1.0)).astype(jnp.float32)

        state = jnp.concatenate((data.qpos, data.qvel))
        bounded = jnp.all(jnp.abs(state[1:]) < 100.0)
        z, angle = data.qpos[1], data.qpos[2]
        if self._task == "walker2d":
            posture = (z > 0.8) & (z < 2.0) & (angle > -1.0) & (angle < 1.0)
        else:
            posture = (z > 0.7) & (angle > -0.2) & (angle < 0.2)
        return (finite & bounded & posture).astype(jnp.float32)

    def _contact_forces(self, data: mjx.Data) -> jax.Array:
        return jnp.clip(data.cfrc_ext, -1.0, 1.0)

    def _contact_cost(self, data: mjx.Data) -> jax.Array:
        if self._task != "ant":
            return jnp.zeros(())
        return 5e-4 * jnp.sum(jnp.square(self._contact_forces(data)))

    def _get_obs(self, data: mjx.Data) -> jax.Array:
        if self._task == "ant":
            obs = jnp.concatenate(
                (data.qpos[2:], data.qvel, self._contact_forces(data).reshape(-1))
            )
        elif self._task in {"walker2d", "hopper"}:
            obs = jnp.concatenate((data.qpos[1:], jnp.clip(data.qvel, -10, 10)))
        else:
            obs = jnp.concatenate((data.qpos[1:], data.qvel))
        return obs

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._spec.action_size

    @property
    def observation_size(self) -> int:
        return self._spec.obs_size

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
