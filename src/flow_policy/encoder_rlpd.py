"""RLPD / high-UTD SAC encoder operating in the decoder latent space."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import optax
from jax import Array

from . import math_utils, networks
from .config_utils import require_config_values


@jdc.pytree_dataclass
class EncoderConfig:
    learning_rate: float | None = None
    critic_learning_rate: float | None = None
    temperature_learning_rate: float | None = None
    discounting: float | None = None
    episode_length: jdc.Static[int | None] = None
    normalize_observations: jdc.Static[bool | None] = None
    num_envs: jdc.Static[int | None] = None
    z_dim: jdc.Static[int | None] = None
    hidden_size: jdc.Static[int | None] = None
    hidden_layers: jdc.Static[int | None] = None
    critic_ensemble_size: jdc.Static[int | None] = None
    critic_subsample_size: jdc.Static[int | None] = None
    target_update_rate: float | None = None
    initial_temperature: float | None = None
    target_entropy: float | None = None
    backup_entropy: jdc.Static[bool | None] = None
    reward_scaling: float | None = None
    reward_bias: float | None = None
    max_grad_norm: float | None = None
    latent_kl_weight: float | None = None
    learn_temperature: jdc.Static[bool | None] = None
    policy_update_period: jdc.Static[int | None] = None
    apply_tanh_in_rollout: jdc.Static[bool | None] = None


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
        require_config_values(
            config,
            allow_none=("target_entropy", "critic_subsample_size"),
        )
        if config.initial_temperature <= 0.0:
            raise ValueError("initial_temperature must be positive")
        if config.latent_kl_weight < 0.0:
            raise ValueError("latent_kl_weight must be non-negative")
        obs_dim = int(env.observation_size)
        actor_key, critic_key, prng = jax.random.split(prng, 3)
        actor_dims = (
            obs_dim,
        ) + (config.hidden_size,) * config.hidden_layers + (config.z_dim,)
        critic_dims = (
            obs_dim + config.z_dim,
        ) + (config.hidden_size,) * config.hidden_layers + (1,)
        actor_params = networks.gaussian_policy_init(actor_key, actor_dims)
        critic_params = tuple(
            networks.mlp_init(key, critic_dims, use_layer_norm=True)
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
        initial_temperature = jnp.asarray(config.initial_temperature)
        log_temperature = jnp.log(jnp.exp(initial_temperature) - 1.0)
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
            self.actor_params if params is None else params,
            self._normalize_obs(obs),
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
        return jax.nn.softplus(self.log_temperature)

    def _target_entropy(self) -> float:
        value = self.config.target_entropy
        if value is not None:
            return float(value)
        # SAC convention: target entropy is the negative action dimension.
        return -float(self.config.z_dim)

    @jax.jit
    def update_observation_stats(self, observations: Array) -> "EncoderState":
        if not self.config.normalize_observations:
            return self
        return jdc.replace(self, obs_stats=self.obs_stats.update(observations))

    @jax.jit
    def evaluate_critic(self, batch: RLPDTransitionBatch) -> dict[str, Array]:
        """Evaluate critic predictions and Bellman targets without updating state."""
        _, action_key, subset_key = jax.random.split(self.prng, 3)
        next_actions, next_log_probs = self._sample_with_params(
            self.actor_params, batch.next_observations, action_key
        )
        target_next_qs = self._critic_values(
            self.target_critic_params, batch.next_observations, next_actions
        )
        if self.config.critic_subsample_size is not None:
            subset_size = min(
                self.config.critic_subsample_size, self.config.critic_ensemble_size
            )
            subset = jax.random.choice(
                subset_key,
                self.config.critic_ensemble_size,
                shape=(subset_size,),
                replace=False,
            )
            target_next_qs = target_next_qs[subset]
        target_q = (
            self.config.reward_scaling * batch.rewards
            + self.config.reward_bias
            + self.config.discounting
            * batch.masks
            * jnp.min(target_next_qs, axis=0)
        )
        if self.config.backup_entropy:
            target_q = (
                target_q
                - jax.lax.stop_gradient(self.temperature) * next_log_probs
            )
        predicted_qs = self._critic_values(
            self.critic_params, batch.observations, batch.actions
        )
        return {
            "predicted_qs": jnp.mean(predicted_qs),
            "target_qs": jnp.mean(target_q),
        }

    @jax.jit
    def update_critic(self, batch: RLPDTransitionBatch) -> tuple["EncoderState", dict[str, Array]]:
        rng, action_key, subset_key = jax.random.split(self.prng, 3)
        next_actions, next_log_probs = self._sample_with_params(
            self.actor_params, batch.next_observations, action_key
        )
        target_next_qs = self._critic_values(
            self.target_critic_params, batch.next_observations, next_actions
        )
        if self.config.critic_subsample_size is not None:
            subset_size = min(
                self.config.critic_subsample_size, self.config.critic_ensemble_size
            )
            subset = jax.random.choice(
                subset_key,
                self.config.critic_ensemble_size,
                shape=(subset_size,),
                replace=False,
            )
            # target_qs = jnp.min(target_qs[subset], axis=0)
            target_next_qs = target_next_qs[subset]
        target_next_min_q = jnp.min(target_next_qs, axis=0)
        target_q = (
                    self.config.reward_scaling * batch.rewards
                    + self.config.reward_bias
                    + self.config.discounting * batch.masks * target_next_min_q
                )
        if self.config.backup_entropy:
            target_q = target_q - jax.lax.stop_gradient(self.temperature) * next_log_probs
        
        target = jax.lax.stop_gradient(target_q)

        def loss_fn(params: Any) -> tuple[Array, tuple[Array, Array, Array]]:
            predicted = self._critic_values(params, batch.observations, batch.actions)
            td_loss = jnp.mean(jnp.square(predicted - target[None, :]))
            return td_loss, (jnp.mean(predicted), jnp.mean(target), td_loss)

        (loss, (predicted_q, target_q_mean, td_loss)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(self.critic_params)
        td_grad_norm = _global_norm(
            jax.grad(lambda p: loss_fn(p)[1][2])(self.critic_params)
        )
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
            "critic_td_grad_norm": td_grad_norm,
            "td_rmse": jnp.sqrt(td_loss + 1e-8),
            "rewards": jnp.mean(batch.rewards),
        }

    @jax.jit
    def update_critic_with_offline_value(
        self,
        batch: RLPDTransitionBatch,
        offline_value_params: networks.MlpWeights,
        offline_value_obs_stats: math_utils.RunningStats,
        bellman_mix: Array,
    ) -> tuple["EncoderState", dict[str, Array]]:
        """Update the critic while interpolating from IQL-V to the RLPD target.

        ``bellman_mix=0`` gives the offline IQL backup
        ``r + gamma * V_offline(s')``.  ``bellman_mix=1`` gives the normal
        online RLPD backup using the target critic and actor-sampled latent.
        The actor and temperature are untouched by this method.
        """
        rng, action_key, subset_key = jax.random.split(self.prng, 3)
        next_actions, next_log_probs = self._sample_with_params(
            self.actor_params, batch.next_observations, action_key
        )
        target_next_qs = self._critic_values(
            self.target_critic_params, batch.next_observations, next_actions
        )
        if self.config.critic_subsample_size is not None:
            subset_size = min(
                self.config.critic_subsample_size, self.config.critic_ensemble_size
            )
            subset = jax.random.choice(
                subset_key,
                self.config.critic_ensemble_size,
                shape=(subset_size,),
                replace=False,
            )
            target_next_qs = target_next_qs[subset]
        online_next_value = jnp.min(target_next_qs, axis=0)

        offline_obs = (
            batch.next_observations - offline_value_obs_stats.mean
        ) / (offline_value_obs_stats.std + 1e-8)
        offline_next_value = networks.value_mlp_fwd(
            offline_value_params, offline_obs
        )
        mixed_next_value = (
            (1.0 - bellman_mix) * offline_next_value
            + bellman_mix * online_next_value
        )
        target_q = (
            self.config.reward_scaling * batch.rewards
            + self.config.reward_bias
            + self.config.discounting * batch.masks * mixed_next_value
        )
        if self.config.backup_entropy:
            target_q = target_q - jax.lax.stop_gradient(
                bellman_mix * self.temperature
            ) * next_log_probs
        target = jax.lax.stop_gradient(target_q)

        def loss_fn(params: Any) -> tuple[Array, tuple[Array, Array, Array]]:
            predicted = self._critic_values(params, batch.observations, batch.actions)
            td_loss = jnp.mean(jnp.square(predicted - target[None, :]))
            return td_loss, (jnp.mean(predicted), jnp.mean(target), td_loss)

        (loss, (predicted_q, target_q_mean, td_loss)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(self.critic_params)
        td_grad_norm = _global_norm(
            jax.grad(lambda p: loss_fn(p)[1][2])(self.critic_params)
        )
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
        predicted_q_mean = jnp.mean(predicted_q)
        td_rmse = jnp.sqrt(td_loss + 1e-8)
        q_scale = jnp.mean(jnp.abs(predicted_q)) + 1e-6
        return state, {
            "critic_loss": loss,
            "predicted_qs": predicted_q,
            "target_qs": target_q_mean,
            "td_rmse": td_rmse,
            "q_scale": q_scale,
            "relative_td_rmse": td_rmse / q_scale,
            "q_gap_ratio": jnp.abs(target_q_mean - predicted_q_mean) / q_scale,
            "critic_grad_norm": _global_norm(grads),
            "critic_td_grad_norm": td_grad_norm,
            "rewards": jnp.mean(batch.rewards),
            "bellman_mix": bellman_mix,
            "offline_next_value": jnp.mean(offline_next_value),
            "online_next_value": jnp.mean(online_next_value),
        }

    @jax.jit
    def update_actor_and_temperature(
        self,
        batch: RLPDTransitionBatch,
        offline_actor_params: networks.MlpWeights | None = None,
        offline_obs_stats: math_utils.RunningStats | None = None,
        offline_actor_kl_weight: float = 5.0,
    ) -> tuple["EncoderState", dict[str, Array]]:
        rng, actor_key, temperature_key = jax.random.split(self.prng, 3)
        temperature = jax.lax.stop_gradient(self.temperature)

        def actor_loss_fn(params: networks.MlpWeights):
            distribution = self._distribution(batch.observations, params)
            actions = distribution.sample(actor_key)
            log_probs = self._log_prob(distribution, actions)
            qs = self._critic_values(self.critic_params, batch.observations, actions)
            q = jnp.mean(qs, axis=0)
            mean = distribution.loc
            std = distribution.scale
            # The frozen FM decoder was trained from a standard-normal latent
            # prior, so retain the target offline-to-online actor objective's
            # KL regularization toward N(0, I).
            prior_kl = 0.5 * jnp.sum(
                jnp.square(mean) + jnp.square(std) - 1.0 - 2.0 * jnp.log(std),
                axis=-1,
            )
            offline_actor_kl = jnp.zeros_like(prior_kl)
            if offline_actor_params is not None and offline_obs_stats is not None:
                offline_obs = (
                    batch.observations - offline_obs_stats.mean
                ) / (offline_obs_stats.std + 1e-8)
                # ``offline_obs`` is already normalized with the frozen
                # offline statistics, so do not normalize it a second time.
                offline_distribution = networks.gaussian_policy_fwd(
                    offline_actor_params, offline_obs
                )
                offline_mean = jax.lax.stop_gradient(offline_distribution.loc)
                offline_std = jax.lax.stop_gradient(offline_distribution.scale)
                offline_actor_kl = jnp.sum(
                    jnp.log(offline_std / std)
                    + (jnp.square(std) + jnp.square(mean - offline_mean))
                    / (2.0 * jnp.square(offline_std))
                    - 0.5,
                    axis=-1,
                )
            loss = jnp.mean(
                temperature * log_probs
                - q
                + self.config.latent_kl_weight * prior_kl
                + offline_actor_kl_weight * offline_actor_kl
            )
            return loss, (
                jnp.mean(-log_probs),
                jnp.mean(q),
                jnp.mean(prior_kl),
                jnp.mean(mean),
                jnp.mean(jnp.abs(mean)),
                jnp.mean(std),
                jnp.mean(jnp.square(std)),
                jnp.min(std),
                jnp.max(std),
                jnp.mean(jnp.linalg.norm(actions, axis=-1)),
                jnp.max(jnp.abs(actions)),
                jnp.mean(offline_actor_kl),
            )

        (actor_loss, actor_aux), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(self.actor_params)
        (
            entropy,
            actor_q,
            latent_prior_kl,
            latent_mean,
            latent_mean_abs,
            latent_std,
            latent_variance,
            latent_std_min,
            latent_std_max,
            latent_norm,
            latent_max_abs,
            offline_actor_kl,
        ) = actor_aux
        actor_optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(self.config.learning_rate),
        )
        actor_updates, actor_opt_state = actor_optimizer.update(
            actor_grads, self.actor_opt_state, self.actor_params
        )
        actor_params = optax.apply_updates(self.actor_params, actor_updates)

        # Temperature is fixed by default while the explicit decoder-prior KL
        # regularizes the actor. Keep the old update path behind a config flag
        # so experiments can opt back into automatic entropy tuning.
        log_temperature = self.log_temperature
        temperature_opt_state = self.temperature_opt_state
        temperature_loss = jnp.zeros(())
        if self.config.learn_temperature:
            target_entropy = self._target_entropy()
            _, temperature_log_probs = self._sample_with_params(
                self.actor_params, batch.next_observations, temperature_key
            )
            temperature_entropy = -jnp.mean(temperature_log_probs)

            def temperature_loss_fn(raw_temperature: Array) -> Array:
                learned_temperature = jax.nn.softplus(raw_temperature)
                return learned_temperature * jax.lax.stop_gradient(
                    temperature_entropy - target_entropy
                )

            temperature_loss, temperature_grads = jax.value_and_grad(
                temperature_loss_fn
            )(self.log_temperature)
            temperature_optimizer = optax.adam(self.config.temperature_learning_rate)
            temperature_updates, temperature_opt_state = temperature_optimizer.update(
                temperature_grads, self.temperature_opt_state, self.log_temperature
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
            "temperature": temperature,
            "temperature_loss": temperature_loss,
            "actor_grad_norm": _global_norm(actor_grads),
            "latent_prior_kl": latent_prior_kl,
            "latent_mean": latent_mean,
            "latent_mean_abs": latent_mean_abs,
            "latent_std": latent_std,
            "latent_variance": latent_variance,
            "latent_std_min": latent_std_min,
            "latent_std_max": latent_std_max,
            "latent_norm": latent_norm,
            "latent_max_abs": latent_max_abs,
            "offline_actor_kl": offline_actor_kl,
        }
