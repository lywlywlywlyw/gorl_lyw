"""Rollout helpers for Encoder with decoder - handles z to action mapping efficiently."""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import traceback
from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax_dataclasses as jdc
import numpy as np
from jax import Array
from jax import numpy as jnp

from . import rollouts
from envs.base_env import State
from envs.robomimic.online_config.env_config import EnvConfig


SUCCESS_REWARD_BONUS = 150.0

def _environment_worker(connection: Any, env_type: type, dataset_path: str) -> None:
    """Own and step one Robomimic environment in a child process."""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env = None
    state = None
    config = EnvConfig().to_dict()
    try:
        env = env_type(dataset_path=dataset_path, reward_shaping=config['dense_reward'])
        while True:
            command, payload = connection.recv()
            try:
                if command == "reset":
                    state = env.reset(payload)
                    result = (np.asarray(state.obs), 0.0, False, False, None)
                elif command == "step":
                    if state is None:
                        raise RuntimeError("Environment must be reset before step().")
                    env_state = env.get_env_state()
                    state = env.step(state, payload)
                    result = (
                        np.asarray(state.obs),
                        float(np.asarray(state.reward)),
                        bool(np.asarray(state.done)),
                        bool(state.info.get("success", False)),
                        env_state,
                    )
                elif command == "close":
                    break
                else:
                    raise ValueError(f"Unknown worker command: {command}")
                connection.send((True, result))
            except Exception:
                connection.send((False, traceback.format_exc()))
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        try:
            connection.send((False, traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        close = getattr(getattr(env, "env", env), "close", None) if env is not None else None
        if callable(close):
            close()
        connection.close()



class EncoderAgentProtocol(Protocol):
    """Protocol for Encoder + Decoder agent state."""
    env: Any

    def sample_z(
        self, obs: Array, prng: Array, deterministic: bool
    ) -> tuple[Array, object]:
        """Sample z from encoder."""
        ...

    def map_z_to_action(
        self, obs: Array, z: Array
    ) -> Array:
        """Map z to action using decoder (deterministic)."""
        ...


@jdc.pytree_dataclass
class EncoderRolloutActionInfo:
    """Policy sampling metadata plus the action executed by the environment."""

    policy_info: Any
    env_action: Array

    @property
    def log_prob(self) -> Array:
        """Preserve the action-info interface expected by PPO training."""
        return self.policy_info.log_prob


@dataclass
class BatchedRolloutStateEncoderFM:
    """CPU environment rollout state for an Encoder + FM decoder policy.

    Robosuite / Robomimic environments are regular Python objects backed by
    MuJoCo and NumPy.  They cannot be transformed with ``jax.vmap`` or executed
    inside ``jax.jit`` / ``jax.lax.scan``.  This class therefore owns one
    independent Python environment per batch element and steps those
    environments in a Python loop.  Observations are stacked into JAX arrays so
    the encoder and decoder still run as batched JAX computations.
    """

    connections: list[Any]
    processes: list[Any]
    env_states: list[Any]
    steps: np.ndarray
    terminated: np.ndarray
    num_envs: int
    prng: Array
    last_transition_env_states: list[Any] | None = None

    @classmethod
    def init(
        cls,
        env: Any,
        prng: Array,
        num_envs: int,
    ) -> "BatchedRolloutStateEncoderFM":
        """Create and reset ``num_envs`` process-isolated CPU environments."""
        if num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")
        prng, reset_prng = jax.random.split(prng, num=2)
        reset_keys = jax.random.split(reset_prng, num=num_envs)
        context = mp.get_context("spawn")
        connections, processes = [], []
        # Spawn imports modules before entering the target. Hide CUDA before
        # start(), otherwise every environment process initializes JAX on GPU.
        worker_environment = {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "JAX_PLATFORM_NAME": "cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        previous_environment = {key: os.environ.get(key) for key in worker_environment}
        saved_stdout, saved_stderr = os.dup(1), os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.environ.update(worker_environment)
            # CUDA plugin discovery occurs before _environment_worker starts.
            # Silence only the spawned interpreters' import-time output; worker
            # failures are still returned explicitly through the Pipe.
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            for _ in range(num_envs):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_environment_worker,
                    args=(child, type(env), env.dataset_path),
                    daemon=True,
                )
                process.start()
                child.close()
                connections.append(parent)
                processes.append(process)
        finally:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)
            os.close(devnull)
            for key, value in previous_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        try:
            for connection, key in zip(connections, reset_keys):
                connection.send(("reset", np.asarray(jax.random.key_data(key))))
            responses = [cls._receive(connection) for connection in connections]
        except BaseException:
            for process in processes:
                process.terminate()
            raise
        env_states = [cls._state(response) for response in responses]
        instance = cls(connections=connections, processes=processes, env_states=env_states, steps=np.zeros(num_envs, dtype=np.int32), terminated=np.zeros(num_envs, dtype=np.bool_), num_envs=num_envs, prng=prng)
        atexit.register(instance.close)
        return instance

    def close(self) -> None:
        """Shut down all workers; safe to call repeatedly."""
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        for connection in self.connections:
            connection.close()
        self.connections = []
        self.processes = []

    def _step_all(self, actions: np.ndarray) -> list[tuple[np.ndarray, float, bool, bool, Any]]:
        for connection, action in zip(self.connections, actions):
            connection.send(("step", action))
        return [self._receive(connection) for connection in self.connections]

    def _reset_indices(self, indices: list[int], keys: list[Array]) -> dict[int, Any]:
        for index, key in zip(indices, keys):
            self.connections[index].send(("reset", np.asarray(jax.random.key_data(key))))
        return {index: self._receive(self.connections[index]) for index in indices}

    @staticmethod
    def _receive(connection: Any) -> tuple[np.ndarray, float, bool, bool, Any]:
        ok, payload = connection.recv()
        if not ok:
            raise RuntimeError(f"Robomimic environment worker failed: {payload}")
        return payload

    @staticmethod
    def _state(response: tuple[np.ndarray, float, bool, bool]) -> State:
        obs, reward, done, success, _ = response
        return State(
            obs=jnp.asarray(obs),
            reward=jnp.asarray(reward),
            done=jnp.asarray(done),
            info={"success": success},
        )

    def rollout(self, agent_state: EncoderAgentProtocol, episode_length: int,
                iterations_per_env: int, auto_reset: bool = True,
                deterministic: bool = False, apply_tanh_in_rollout: bool = True):
        """Collect transitions with parallel CPU env steps and batched JAX inference."""
        if iterations_per_env < 1:
            raise ValueError("iterations_per_env must be positive.")
        transition_steps, prng = [], self.prng
        transition_env_states: list[Any] = []
        for _ in range(iterations_per_env):
            obs = jnp.stack([jnp.asarray(state.obs) for state in self.env_states])
            prng_z, prng = jax.random.split(prng)
            z, z_info = agent_state.sample_z(obs, prng_z, deterministic=deterministic)
            # The encoder outputs latent z. The frozen decoder maps z to the
            # environment action directly; do not tanh decoder output again.
            env_action = agent_state.map_z_to_action(obs, z)
            responses = self._step_all(np.asarray(jax.device_get(env_action)))
            transition_env_states.extend(response[4] for response in responses)
            next_states, transition_next_states = [], []
            rewards, truncations, discounts = [], [], []
            reset_indices, reset_keys = [], []
            for env_index, (env_state, response) in enumerate(zip(self.env_states, responses)):
                if not auto_reset and self.terminated[env_index]:
                    next_state = env_state.replace(reward=jnp.asarray(0.0), done=jnp.asarray(True))
                    next_step, done = self.steps[env_index], True
                else:
                    next_state = self._state(response)
                    next_step = self.steps[env_index] + 1
                    success = bool(next_state.info.get("success", False))
                    reached_episode_limit = next_step >= episode_length
                    done = success or reached_episode_limit
                    reward = float(np.asarray(next_state.reward))
                    if success:
                        reward += SUCCESS_REWARD_BONUS
                    next_state = next_state.replace(
                        reward=jnp.asarray(reward, dtype=jnp.float32),
                        done=jnp.asarray(done),
                    )
                # The requested online semantics treat both success and the
                # configured episode-length limit as true terminal transitions,
                # rather than bootstrappable time-limit truncations.
                truncated = False
                transition_next_states.append(next_state)
                rewards.append(next_state.reward); truncations.append(truncated)
                discounts.append(0.0 if done else 1.0)
                if auto_reset and (done or truncated):
                    prng, reset_prng = jax.random.split(prng)
                    reset_indices.append(env_index); reset_keys.append(reset_prng)
                    next_states.append(None)
                    self.steps[env_index] = 0; self.terminated[env_index] = False
                else:
                    next_states.append(next_state); self.steps[env_index] = next_step
                    self.terminated[env_index] = done or truncated
            for env_index, response in self._reset_indices(reset_indices, reset_keys).items():
                next_states[env_index] = self._state(response)
            transition_steps.append(rollouts.TransitionStruct(
                obs=obs, next_obs=jnp.stack([state.obs for state in transition_next_states]),
                action=z,
                action_info=EncoderRolloutActionInfo(
                    policy_info=z_info,
                    env_action=env_action,
                ),
                reward=jnp.asarray(rewards, dtype=jnp.float32),
                truncation=jnp.asarray(truncations, dtype=jnp.float32),
                discount=jnp.asarray(discounts, dtype=jnp.float32)))
            self.env_states = next_states
        self.prng = prng
        self.last_transition_env_states = transition_env_states
        return self, jax.tree.map(lambda *xs: jnp.stack(xs), *transition_steps)

    def rollout_with_actions(self, agent, episode_length: int, iterations_per_env: int,
                             apply_tanh_in_rollout: bool = True):
        """Collect decoder actions from process-isolated environments."""
        states, actions, rewards, prng = [], [], [], self.prng
        for _ in range(iterations_per_env):
            obs = jnp.stack([state.obs for state in self.env_states])
            prng_sample, prng = jax.random.split(prng)
            z, _ = agent.sample_z(obs, prng_sample, deterministic=False)
            # Decoder output is already the environment action.
            env_action = agent.map_z_to_action(obs, z)
            responses = self._step_all(np.asarray(jax.device_get(env_action)))
            next_states, step_rewards, reset_indices, reset_keys = [], [], [], []
            for env_index, response in enumerate(responses):
                next_state = self._state(response); next_step = self.steps[env_index] + 1
                success = bool(next_state.info.get("success", False))
                done = success or next_step >= episode_length
                reward = float(np.asarray(next_state.reward))
                if success:
                    reward += SUCCESS_REWARD_BONUS
                next_state = next_state.replace(
                    reward=jnp.asarray(reward, dtype=jnp.float32),
                    done=jnp.asarray(done),
                )
                step_rewards.append(next_state.reward)
                if done:
                    prng, reset_prng = jax.random.split(prng)
                    reset_indices.append(env_index); reset_keys.append(reset_prng)
                    next_states.append(None); self.steps[env_index] = 0
                else:
                    next_states.append(next_state); self.steps[env_index] = next_step
                self.terminated[env_index] = False
            for env_index, response in self._reset_indices(reset_indices, reset_keys).items():
                next_states[env_index] = self._state(response)
            # Store the exact decoder action executed by the environment. No
            # extra tanh is applied before env.step or decoder training.
            states.append(obs); actions.append(env_action)
            rewards.append(jnp.asarray(step_rewards, dtype=jnp.float32)); self.env_states = next_states
        self.prng = prng
        return self, jnp.stack(states), jnp.stack(actions), jnp.stack(rewards)

# Alias for Diffusion - same implementation, different name for clarity
BatchedRolloutStateEncoderDiffusion = BatchedRolloutStateEncoderFM


def eval_policy_encoder_fm(
    agent_state: EncoderAgentProtocol,
    prng: Array,
    num_envs: int,
    max_episode_length: int,
    apply_tanh_in_rollout: bool = True,
) -> rollouts.EvalOutputs:
    """Run policy evaluation for Encoder with FM decoder."""
    rollout_state = BatchedRolloutStateEncoderFM.init(
        agent_state.env, prng, num_envs
    )

    try:
        _, transitions = rollout_state.rollout(
            agent_state, episode_length=max_episode_length,
            iterations_per_env=max_episode_length, auto_reset=False,
            deterministic=True, apply_tanh_in_rollout=apply_tanh_in_rollout)
    finally:
        rollout_state.close()
    valid_mask = transitions.discount > 0.0

    rewards = jnp.sum(transitions.reward, axis=0)
    steps = jnp.sum(valid_mask, axis=0)

    scalar_metrics = {
        "reward_mean": jnp.mean(rewards),
        "reward_min": jnp.min(rewards),
        "reward_max": jnp.max(rewards),
        "reward_std": jnp.std(rewards),
        "steps_mean": jnp.mean(steps),
        "steps_min": jnp.min(steps),
        "steps_max": jnp.max(steps),
        "steps_std": jnp.std(steps),
    }

    histogram_metrics = {
        "reward": rewards.flatten(),
        "steps": steps.flatten(),
    }

    return rollouts.EvalOutputs(
        scalar_metrics=scalar_metrics,
        histogram_metrics=histogram_metrics,
        actions=transitions.action,
        action_timestep_mask=valid_mask,
    )


# Alias for Diffusion
eval_policy_encoder_diffusion = eval_policy_encoder_fm
