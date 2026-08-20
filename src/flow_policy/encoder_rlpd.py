"""RLPD / high-UTD SAC encoder operating in the decoder latent space."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import optax
from jax import Array

from . import math_utils, networks


@jdc.pytree_dataclass
class EncoderConfig:
    learning_rate: float
    critic_learning_rate: float
    temperature_learning_rate: float
    discounting: float
    episode_length: jdc.Static[int]
    normalize_observations: jdc.Static[bool]
    num_envs: jdc.Static[int]
    z_dim: jdc.Static[int]
    hidden_size: jdc.Static[int] = 256
    hidden_layers: jdc.Static[int] = 2
    critic_ensemble_size: jdc.Static[int] = 10
    critic_subsample_size: jdc.Static[int] = 2
    target_update_rate: float = 0.005
    initial_temperature: float = 1.0
    target_entropy: float | None = None
    backup_entropy: jdc.Static[bool] = True
    reward_scaling: float = 1.0
    reward_bias: float = 0.0
    max_grad_norm: float = 10.0
    policy_update_period: jdc.Static[int] = 20
    apply_tanh_in_rollout: jdc.Static[bool] = True


class RLPDActionInfo(NamedTuple):
    raw_action: Array
    log_prob: Array


class RLPDTransitionBatch(NamedTuple):
    observations: Array
    actions: Array
    rewards: Array
    next_observations: Array
    masks: Array


def _tree_soft_update(target: Any, source: Any, tau: float) -> Any:
    return jax.tree.map(lambda t, s: (1.0 - tau) * t + tau * s, target, source)


def _global_norm(tree: Any) -> Array:
    return optax.global_norm(tree)


@jdc.pytree_dataclass
class EncoderState:
    actor_params: networks.MlpWeights
    critic_params: tuple[networks.MlpWeights, ...]
    target_critic_params: tuple[networks.MlpWeights, ...]
    log_temperature: Array
    actor_opt_state: optax.OptState
    critic_opt_state: optax.OptState
    temperature_opt_state: optax.OptState
    obs_stats: math_utils.RunningStats
    prng: Array
    steps: Array
    config: jdc.Static[EncoderConfig]
    env: jdc.Static[Any]

    @staticmethod
    def init(prng: Array, env: Any, config: EncoderConfig) -> "EncoderState":
        obs_dim = int(env.observation_size)
        actor_key, critic_key, prng = jax.random.split(prng, 3)
        actor_dims = (obs_dim,) + (config.hidden_size,) * config.hidden_layers + (
            2 * config.z_dim,
        )
        critic_dims = (
            obs_dim + config.z_dim,
        ) + (config.hidden_size,) * config.hidden_layers + (1,)
        actor_params = networks.mlp_init(actor_key, actor_dims)
        critic_params = tuple(
            networks.mlp_init(key, critic_dims)
            for key in jax.random.split(critic_key, config.critic_ensemble_size)
        )
        actor_optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate),
        )
        critic_optimizer = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.critic_learning_rate),
        )
        temperature_optimizer = optax.adam(config.temperature_learning_rate)
        log_temperature = jnp.log(jnp.asarray(config.initial_temperature))
        return EncoderState(
            actor_params=actor_params,
            critic_params=critic_params,
            target_critic_params=critic_params,
            log_temperature=log_temperature,
            actor_opt_state=actor_optimizer.init(actor_params),
            critic_opt_state=critic_optimizer.init(critic_params),
            temperature_opt_state=temperature_optimizer.init(log_temperature),
            obs_stats=math_utils.RunningStats.init((obs_dim,)),
            prng=prng,
            steps=jnp.zeros((), dtype=jnp.int32),
            config=config,
            env=env,
        )

    def _normalize_obs(self, obs: Array) -> Array:
        if not self.config.normalize_observations:
            return obs
        return (obs - self.obs_stats.mean) / (self.obs_stats.std + 1e-8)

    def _distribution(self, obs: Array, params: networks.MlpWeights | None = None):
        return networks.gaussian_policy_fwd(
            self.actor_params if params is None else params, self._normalize_obs(obs)
        )

    @staticmethod
    def _squashed_log_prob(distribution: Any, raw_action: Array) -> Array:
        log_prob = jnp.sum(distribution.log_prob(raw_action), axis=-1)
        correction = jnp.sum(math_utils.tanh_log_det_jacobian(raw_action), axis=-1)
        return log_prob - correction

    @staticmethod
    def _log_prob(distribution: Any, action: Array) -> Array:
        return jnp.sum(distribution.log_prob(action), axis=-1)

    def sample_z(
        self, obs: Array, prng: Array, deterministic: bool
    ) -> tuple[Array, RLPDActionInfo]:
        distribution = self._distribution(obs)
        z = distribution.loc if deterministic else distribution.sample(prng)
        
        # log_prob = self._squashed_log_prob(distribution, z)
        log_prob = self._log_prob(distribution, z)
        return z, RLPDActionInfo(raw_action=z, log_prob=log_prob)

    def _sample_with_params(
        self, params: networks.MlpWeights, obs: Array, prng: Array
    ) -> tuple[Array, Array]:
        distribution = self._distribution(obs, params)
        raw_action = distribution.sample(prng)
        # return jnp.tanh(raw_action), self._squashed_log_prob(distribution, raw_action)
        return raw_action, self._log_prob(distribution, raw_action)#self._squashed_log_prob(distribution, raw_action)

    def _critic_values(self, params: Any, obs: Array, actions: Array) -> Array:
        obs_norm = self._normalize_obs(obs)
        return jnp.stack(
            [networks.q_mlp_fwd(member, obs_norm, actions) for member in params]
        )

    @property
    def temperature(self) -> Array:
        return jnp.exp(self.log_temperature)

    def _target_entropy(self) -> float:
        value = self.config.target_entropy
        return -float(self.config.z_dim) if value is None else float(value)

    @jax.jit
    def update_observation_stats(self, observations: Array) -> "EncoderState":
        if not self.config.normalize_observations:
            return self
        return jdc.replace(self, obs_stats=self.obs_stats.update(observations))

    @jax.jit
    def update_critic(self, batch: RLPDTransitionBatch) -> tuple["EncoderState", dict[str, Array]]:
        rng, action_key, subset_key = jax.random.split(self.prng, 3)
        next_actions, next_log_probs = self._sample_with_params(
            self.actor_params, batch.next_observations, action_key
        )
        target_qs = self._critic_values(
            self.target_critic_params, batch.next_observations, next_actions
        )
        subset_size = min(
            self.config.critic_subsample_size, self.config.critic_ensemble_size
        )
        subset = jax.random.choice(
            subset_key,
            self.config.critic_ensemble_size,
            shape=(subset_size,),
            replace=False,
        )
        target_q = jnp.min(target_qs[subset], axis=0)
        if self.config.backup_entropy:
            target_q = target_q - jax.lax.stop_gradient(self.temperature) * next_log_probs
        target = (
            self.config.reward_scaling * batch.rewards
            + self.config.reward_bias
            + self.config.discounting * batch.masks * target_q
        )
        target = jax.lax.stop_gradient(target)

        def loss_fn(params: Any) -> tuple[Array, tuple[Array, Array]]:
            predicted = self._critic_values(params, batch.observations, batch.actions)
            loss = jnp.mean(jnp.square(predicted - target[None, :]))
            return loss, (jnp.mean(predicted), jnp.mean(target))

        (loss, (predicted_q, target_q_mean)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(self.critic_params)
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(self.config.critic_learning_rate),
        )
        updates, opt_state = optimizer.update(
            grads, self.critic_opt_state, self.critic_params
        )
        critic_params = optax.apply_updates(self.critic_params, updates)
        state = jdc.replace(
            self,
            critic_params=critic_params,
            target_critic_params=_tree_soft_update(
                self.target_critic_params,
                critic_params,
                self.config.target_update_rate,
            ),
            critic_opt_state=opt_state,
            prng=rng,
            steps=self.steps + 1,
        )
        return state, {
            "critic_loss": loss,
            "predicted_qs": predicted_q,
            "target_qs": target_q_mean,
            "critic_grad_norm": _global_norm(grads),
            "rewards": jnp.mean(batch.rewards),
        }

    @jax.jit
    def update_actor_and_temperature(
        self, batch: RLPDTransitionBatch
    ) -> tuple["EncoderState", dict[str, Array]]:
        rng, actor_key = jax.random.split(self.prng)
        temperature = jax.lax.stop_gradient(self.temperature)

        def actor_loss_fn(params: networks.MlpWeights):
            actions, log_probs = self._sample_with_params(
                params, batch.observations, actor_key
            )
            qs = self._critic_values(self.critic_params, batch.observations, actions)
            q = jnp.mean(qs, axis=0)
            loss = jnp.mean(temperature * log_probs - q)
            return loss, (jnp.mean(-log_probs), jnp.mean(q), log_probs)

        (actor_loss, (entropy, actor_q, log_probs)), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(self.actor_params)
        actor_optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(self.config.learning_rate),
        )
        actor_updates, actor_opt_state = actor_optimizer.update(
            actor_grads, self.actor_opt_state, self.actor_params
        )
        actor_params = optax.apply_updates(self.actor_params, actor_updates)
        target_entropy = self._target_entropy()

        def temperature_loss_fn(log_temperature: Array) -> Array:
            return -jnp.mean(
                log_temperature
                * jax.lax.stop_gradient(log_probs + target_entropy)
            )

        temperature_loss, temperature_grads = jax.value_and_grad(
            temperature_loss_fn
        )(self.log_temperature)
        temperature_optimizer = optax.adam(self.config.temperature_learning_rate)
        temperature_updates, temperature_opt_state = temperature_optimizer.update(
            temperature_grads,
            self.temperature_opt_state,
            self.log_temperature,
        )
        log_temperature = optax.apply_updates(
            self.log_temperature, temperature_updates
        )
        state = jdc.replace(
            self,
            actor_params=actor_params,
            log_temperature=log_temperature,
            actor_opt_state=actor_opt_state,
            temperature_opt_state=temperature_opt_state,
            prng=rng,
        )
        return state, {
            "actor_loss": actor_loss,
            "actor_q": actor_q,
            "entropy": entropy,
            "temperature": jnp.exp(log_temperature),
            "temperature_loss": temperature_loss,
            "actor_grad_norm": _global_norm(actor_grads),
        }
