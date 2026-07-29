"""One-step MeanFlow decoder policy.

This mirrors decoder_fm.py's public interface while replacing multi-step
Euler sampling with the MeanFlow identity used by MP1.
"""

from __future__ import annotations

import jax
import jax_dataclasses as jdc
import optax
from jax import Array
from jax import numpy as jnp

from flow_policy.networks import MlpWeights
from . import math_utils, networks


@jdc.pytree_dataclass
class Decoder1StepFMConfig:
    """Configuration for one-step MeanFlow decoding."""

    # Kept for checkpoint compatibility with FM-style pipelines.
    flow_steps: jdc.Static[int] = 1
    timestep_embed_dim: jdc.Static[int] = 8

    hidden_dims: jdc.Static[tuple[int, ...]] = (64, 64, 64, 64)
    policy_output_scale: float = 1.0

    learning_rate: float = 3e-4
    batch_size: jdc.Static[int] = 8192
    num_epochs: jdc.Static[int] = 50
    n_samples_per_action: jdc.Static[int] = 8

    normalize_observations: jdc.Static[bool] = True
    normalize_actions: jdc.Static[bool] = True

    # MP1/MeanFlow sampling of interval endpoints.
    flow_ratio: float = 0.5
    time_dist: jdc.Static[str] = "lognorm"
    lognorm_mu: float = -0.4
    lognorm_sigma: float = 1.0
    adaptive_loss_gamma: float = 0.5
    adaptive_loss_c: float = 1e-3
    guidance_scale: float = 2.0
    dispersive_loss_weight: float = 0.5
    dispersive_tau: float = 1.0

    feather_std: float = 0.0


@jdc.pytree_dataclass
class Decoder1StepFMState:
    """One-step MeanFlow model state."""

    config: Decoder1StepFMConfig
    params: MlpWeights
    obs_stats: math_utils.RunningStats
    action_stats: math_utils.RunningStats
    opt: jdc.Static[optax.GradientTransformation]
    opt_state: optax.OptState
    prng: Array
    steps: Array

    @staticmethod
    def init(
        prng: Array,
        obs_dim: int,
        action_dim: int,
        config: Decoder1StepFMConfig,
    ) -> Decoder1StepFMState:
        prng0, prng1 = jax.random.split(prng)

        input_dim = obs_dim + action_dim + 2 * config.timestep_embed_dim
        layer_dims = (input_dim,) + config.hidden_dims + (action_dim,)
        meanflow_net = networks.mlp_init(prng0, layer_dims)

        opt = optax.adam(config.learning_rate)

        return Decoder1StepFMState(
            config=config,
            params=meanflow_net,
            obs_stats=math_utils.RunningStats.init((obs_dim,)),
            action_stats=math_utils.RunningStats.init((action_dim,)),
            opt=opt,
            opt_state=opt.init(meanflow_net),
            prng=prng1,
            steps=jnp.zeros((), dtype=jnp.int32),
        )

    def embed_timestep(self, t: Array) -> Array:
        """Embed a scalar timestep with the same sinusoidal basis as FM."""
        assert t.shape[-1] == 1, f"Expected (..., 1), got {t.shape}"
        freqs = jnp.arange(self.config.timestep_embed_dim // 2)
        scaled_t = t * (2 ** freqs[None, :])
        return jnp.concatenate([jnp.cos(scaled_t), jnp.sin(scaled_t)], axis=-1)

    def meanflow_forward(
        self,
        obs_norm: Array,
        x_t: Array,
        t_embed: Array,
        r_embed: Array,
    ) -> Array:
        """Predict the interval-averaged velocity u(x_t, t, r)."""
        mean_velocity = networks.flow_mlp_fwd(
            self.params,
            obs_norm,
            x_t,
            t_embed,
            r_embed,
        )
        return mean_velocity * self.config.policy_output_scale

    def _normalize_obs(self, obs: Array) -> Array:
        if self.config.normalize_observations:
            return (obs - self.obs_stats.mean) / (self.obs_stats.std + 1e-8)
        return obs

    def _normalize_action(self, action: Array) -> Array:
        if self.config.normalize_actions:
            return (action - self.action_stats.mean) / (
                self.action_stats.std + 1e-8
            )
        return action

    def _unnormalize_action(self, action: Array) -> Array:
        if self.config.normalize_actions:
            return action * (self.action_stats.std + 1e-8) + self.action_stats.mean
        return action

    def sample_action(
        self,
        obs: Array,
        prng: Array,
        deterministic: bool = False,
    ) -> Array:
        """Sample an action with one network function evaluation."""
        obs_norm = self._normalize_obs(obs)

        single_obs = obs.ndim == 1
        if single_obs:
            obs_norm = obs_norm[None, :]

        (*batch_dims, _) = obs_norm.shape
        action_dim = self.params[-1][0].shape[-1]

        prng_sample, prng_feather = jax.random.split(prng, 2)
        z = jax.random.normal(prng_sample, (*batch_dims, action_dim))

        t = jnp.ones((*batch_dims, 1))
        r = jnp.zeros((*batch_dims, 1))
        action_norm = z - self.meanflow_forward(
            obs_norm,
            z,
            self.embed_timestep(t),
            self.embed_timestep(r),
        )
        action = self._unnormalize_action(action_norm)

        if not deterministic:
            action = action + (
                jax.random.normal(prng_feather, (*batch_dims, action_dim))
                * self.config.feather_std
            )

        if single_obs:
            action = action.squeeze(0)

        return action

    def sample_action_from_z(
        self,
        obs: Array,
        z: Array,
        prng: Array,
        deterministic: bool = True,
    ) -> Array:
        """Decode an externally supplied latent z in one step."""
        obs_norm = self._normalize_obs(obs)

        single_obs = obs.ndim == 1
        if single_obs:
            obs_norm = obs_norm[None, :]
            z = z[None, :]

        (*batch_dims, _) = obs_norm.shape
        t = jnp.ones((*batch_dims, 1))
        r = jnp.zeros((*batch_dims, 1))
        action_norm = z - self.meanflow_forward(
            obs_norm,
            z,
            self.embed_timestep(t),
            self.embed_timestep(r),
        )
        action = self._unnormalize_action(action_norm)

        if not deterministic:
            action_dim = self.params[-1][0].shape[-1]
            action = action + (
                jax.random.normal(prng, (*batch_dims, action_dim))
                * self.config.feather_std
            )

        if single_obs:
            action = action.squeeze(0)

        return action

    def sample_t_r(
        self,
        prng: Array,
        batch_size: int,
    ) -> tuple[Array, Array]:
        """Sample one (t, r) pair per batch item, matching MeanPolicy."""
        prng_time, prng_flow = jax.random.split(prng)

        if self.config.time_dist == "uniform":
            samples = jax.random.uniform(prng_time, (batch_size, 2))
        elif self.config.time_dist == "lognorm":
            normal_samples = (
                jax.random.normal(prng_time, (batch_size, 2))
                * self.config.lognorm_sigma
                + self.config.lognorm_mu
            )
            samples = jax.nn.sigmoid(normal_samples)
        else:
            raise ValueError(f"Unsupported time_dist: {self.config.time_dist}")

        t = jnp.maximum(samples[..., 0], samples[..., 1])
        r = jnp.minimum(samples[..., 0], samples[..., 1])

        flow_mask = (
            jax.random.uniform(prng_flow, (batch_size,))
            < self.config.flow_ratio
        )
        r = jnp.where(flow_mask, t, r)
        return t[:, None], r[:, None]

    def adaptive_l2_loss(self, error: Array) -> Array:
        """MP1 adaptive L2 loss, reduced over action dimensions."""
        delta_sq = jnp.mean(error**2, axis=-1)
        p = 1.0 - self.config.adaptive_loss_gamma
        weight = jax.lax.stop_gradient(
            1.0 / jnp.power(delta_sq + self.config.adaptive_loss_c, p)
        )
        return weight * delta_sq

    def dispersive_loss(self, prediction: Array) -> Array:
        """Encourage predictions within the batch to remain dispersed."""
        differences = prediction[:, None, :] - prediction[None, :, :]
        distances = jnp.sum(differences**2, axis=-1)
        distances = distances / jnp.maximum(jnp.max(distances), 1e-8)
        return jnp.log(jnp.mean(jnp.exp(-distances / self.config.dispersive_tau)))

    def compute_meanflow_loss(
        self,
        obs_norm: Array,
        action: Array,
        eps: Array,
        t: Array,
        r: Array,
    ) -> Array:
        """Compute one-step MeanFlow identity loss.

        The path is identical to decoder_fm.py: t=1 is noise and t=0 is action.
        The learned u approximates the interval-averaged velocity from r to t,
        so inference recovers x_0 by x_1 - u(x_1, 1, 0).
        """
        assert action.ndim == eps.ndim == 2
        assert eps.shape == action.shape
        assert t.shape == r.shape == (action.shape[0], 1)

        action_norm = self._normalize_action(action)
        x_t = t * eps + (1.0 - t) * action_norm
        v = eps - action_norm

        def model_fn(z_arg: Array, t_arg: Array, r_arg: Array) -> Array:
            return self.meanflow_forward(
                obs_norm,
                z_arg,
                self.embed_timestep(t_arg),
                self.embed_timestep(r_arg),
            )

        u_t = jax.lax.stop_gradient(
            model_fn(x_t, t, t)
        )
        v_hat = (
            self.config.guidance_scale * v
            + (1.0 - self.config.guidance_scale) * u_t
        )
        u, dudt = jax.jvp(
            model_fn,
            (x_t, t, r),
            (v_hat, jnp.ones_like(t), jnp.zeros_like(r)),
        )
        u_target = jax.lax.stop_gradient(v_hat - (t - r) * dudt)
        meanflow_loss = jnp.mean(self.adaptive_l2_loss(u - u_target))
        dis_loss = self.dispersive_loss(u)
        loss = meanflow_loss + self.config.dispersive_loss_weight * dis_loss
        return loss, meanflow_loss, dis_loss

    def train_step(
        self,
        batch_obs: Array,
        batch_actions: Array,
    ) -> tuple[Decoder1StepFMState, dict[str, Array]]:
        batch_size = batch_obs.shape[0]
        action_dim = batch_actions.shape[1]

        obs_norm = self._normalize_obs(batch_obs)

        prng_eps, prng_tr, self_prng = jax.random.split(self.prng, 3)
        eps = jax.random.normal(prng_eps, (batch_size, action_dim))
        t, r = self.sample_t_r(prng_tr, batch_size)

        def loss_fn(params):
            state_with_params = jdc.replace(self, params=params)
            loss, meanflow_loss, dis_loss = state_with_params.compute_meanflow_loss(
                obs_norm,
                batch_actions,
                eps,
                t,
                r,
            )
            metrics = {
                "loss": loss,
                "meanflow_loss": meanflow_loss,
                "dis_loss": dis_loss,
                "t_mean": jnp.mean(t),
                "r_mean": jnp.mean(r),
            }
            return loss, metrics

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            self.params
        )
        del loss

        updates, new_opt_state = self.opt.update(grads, self.opt_state)
        new_params = optax.apply_updates(self.params, updates)

        with jdc.copy_and_mutate(self) as new_state:
            new_state.params = new_params
            new_state.opt_state = new_opt_state
            new_state.prng = self_prng
            new_state.steps = self.steps + 1

        return new_state, metrics
