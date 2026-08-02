"""Rollout helpers for Encoder with decoder - handles z to action mapping efficiently."""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Any, Protocol

import jax
# import jax_dataclasses as jdc
import numpy as np
from jax import Array
from jax import numpy as jnp

from . import rollouts


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

    envs: list[Any]
    env_states: list[Any]
    steps: np.ndarray
    terminated: np.ndarray
    num_envs: int
    prng: Array

    @staticmethod
    def _make_env(template_env: Any) -> Any:
        """Create an independent environment from a configuration template."""
        return type(template_env)(dataset_path=template_env.dataset_path)
    
    @classmethod
    def init(
        cls,
        env: Any,
        prng: Array,
        num_envs: int,
    ) -> "BatchedRolloutStateEncoderFM":
        """Create and reset ``num_envs`` independent CPU environments."""
        if num_envs < 1:
            raise ValueError(f"num_envs must be positive, got {num_envs}.")
        if num_envs > 16:
            warnings.warn(
                f"Creating {num_envs} Robosuite environments in one process is "
                "likely to be slow and memory intensive. Start with 1, 8, or 16; "
                "use a multiprocessing vector environment for larger batches.",
                RuntimeWarning,
                stacklevel=2,
            )

        prng, reset_prng = jax.random.split(prng, num=2)
        reset_keys = jax.random.split(reset_prng, num=num_envs)
        envs = [cls._make_env(env) for _ in range(num_envs)]
        env_states = [cpu_env.reset(key) for cpu_env, key in zip(envs, reset_keys)]
        return cls(
            envs=envs,
            env_states=env_states,
            steps=np.zeros(num_envs, dtype=np.int32),
            terminated=np.zeros(num_envs, dtype=np.bool_),
            num_envs=num_envs,
            prng=prng,
        )

    def rollout(
        self,
        agent_state: EncoderAgentProtocol,
        episode_length: int,
        iterations_per_env: int,
        auto_reset: bool = True,
        deterministic: bool = False,
        apply_tanh_in_rollout: bool = True,
    ) -> tuple["BatchedRolloutStateEncoderFM", rollouts.TransitionStruct]:
        """Collect transitions with CPU env steps and batched JAX inference."""
        if iterations_per_env < 1:
            raise ValueError("iterations_per_env must be positive.")

        transition_steps = []
        prng = self.prng

        for _ in range(iterations_per_env):
            obs = jnp.stack([jnp.asarray(state.obs) for state in self.env_states])
            prng_z, prng = jax.random.split(prng)

            # These are the only policy-side operations in the rollout. They
            # operate on the complete batch as JAX arrays.
            z, z_info = agent_state.sample_z(obs, prng_z, deterministic=deterministic)
            action = agent_state.map_z_to_action(obs, z)
            env_action = jnp.tanh(action) if apply_tanh_in_rollout else action

            # Moving actions to NumPy is the explicit JAX -> CPU environment
            # boundary and synchronizes any pending accelerator computation.
            next_states = []
            transition_next_states = []
            rewards = []
            truncations = []
            discounts = []

            for env_index, (cpu_env, env_state) in enumerate(
                zip(self.envs, self.env_states)
            ):
                if not auto_reset and self.terminated[env_index]:
                    next_state = env_state.replace(
                        reward=jnp.asarray(0.0), done=jnp.asarray(True)
                    )
                    next_step = self.steps[env_index]
                    truncated = next_step >= episode_length
                    done = True
                else:
                    next_state = cpu_env.step(env_state, env_action[env_index])
                    next_step = self.steps[env_index] + 1
                    truncated = next_step >= episode_length
                    done = bool(np.asarray(next_state.done))

                transition_next_states.append(next_state)
                rewards.append(next_state.reward)
                truncations.append(truncated)
                discounts.append(0.0 if done else 1.0)

                if auto_reset and (done or truncated):
                    prng, reset_prng = jax.random.split(prng)
                    next_states.append(cpu_env.reset(reset_prng))
                    self.steps[env_index] = 0
                    self.terminated[env_index] = False
                else:
                    next_states.append(next_state)
                    self.steps[env_index] = next_step
                    self.terminated[env_index] = done or truncated

            transition_steps.append(
                rollouts.TransitionStruct(
                    obs=obs,
                    next_obs=jnp.stack(
                        [jnp.asarray(state.obs) for state in transition_next_states]
                    ),
                    action=z,  # PPO is trained in latent z space, not action space.
                    action_info=z_info,
                    reward=jnp.asarray(rewards, dtype=jnp.float32),
                    truncation=jnp.asarray(truncations, dtype=jnp.float32),
                    discount=jnp.asarray(discounts, dtype=jnp.float32),
                )
            )
            self.env_states = next_states

        self.prng = prng
        transitions = jax.tree.map(lambda *xs: jnp.stack(xs), *transition_steps)
        return self, transitions

    def rollout_with_actions(
        self, agent, episode_length: int, iterations_per_env: int,
        apply_tanh_in_rollout: bool = True
    ) -> tuple[BatchedRolloutStateEncoderFM, Array, Array, Array]:
        """Rollout that returns actual actions (not z) for data collection."""
        states = []
        actions = []
        rewards = []
        prng = self.prng

        for _ in range(iterations_per_env):
            obs = jnp.stack([jnp.asarray(state.obs) for state in self.env_states])
            prng_sample, prng = jax.random.split(prng)
            z, _ = agent.sample_z(obs, prng_sample, deterministic=False)
            action = agent.map_z_to_action(obs, z)
            env_action = jnp.tanh(action) if apply_tanh_in_rollout else action
            
            next_states = []
            step_rewards = []
            for env_index, (cpu_env, env_state) in enumerate(
                zip(self.envs, self.env_states)
            ):
                next_state = cpu_env.step(env_state, env_action[env_index])
                next_step = self.steps[env_index] + 1
                done = bool(np.asarray(next_state.done))
                truncated = next_step >= episode_length
                step_rewards.append(next_state.reward)

                if done or truncated:
                    prng, reset_prng = jax.random.split(prng)
                    next_states.append(cpu_env.reset(reset_prng))
                    self.steps[env_index] = 0
                else:
                    next_states.append(next_state)
                    self.steps[env_index] = next_step
                self.terminated[env_index] = False

            states.append(obs)
            actions.append(action)
            rewards.append(jnp.asarray(step_rewards, dtype=jnp.float32))
            self.env_states = next_states

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

    _, transitions = rollout_state.rollout(
        agent_state,
        episode_length=max_episode_length,
        iterations_per_env=max_episode_length,
        auto_reset=False,
        deterministic=True,
        apply_tanh_in_rollout=apply_tanh_in_rollout,
    )
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
