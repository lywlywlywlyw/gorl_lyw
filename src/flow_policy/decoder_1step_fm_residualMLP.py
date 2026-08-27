"""One-step MeanFlow decoder implemented as a conditional residual MLP.

The decoder operates on a single action vector rather than an action horizon.
Consequently, an MLP is a better inductive bias than the 1D U-Net used by the
sequence decoder: it avoids repeatedly down/up-sampling a horizon of length one
while retaining strong state/time conditioning through FiLM residual blocks.
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax_dataclasses as jdc
import optax
from jax import Array
from jax import numpy as jnp



class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding for continuous MeanFlow times."""

    dim: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        half_dim = self.dim // 2
        scale = jnp.log(10000.0) / (half_dim - 1)
        frequencies = jnp.exp(jnp.arange(half_dim) * -scale)
        embedding = x[:, None] * frequencies[None, :]
        return jnp.concatenate([jnp.sin(embedding), jnp.cos(embedding)], axis=-1)


class TimeEncoder(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, time: Array) -> Array:
        x = SinusoidalPosEmb(self.dim)(time)
        x = nn.Dense(self.dim * 4)(x)
        x = jax.nn.mish(x)
        return nn.Dense(self.dim)(x)


class ConditionalResidualMLPBlock(nn.Module):
    """Pre-normalized residual MLP block with FiLM conditioning."""

    hidden_dim: int
    expansion: int

    @nn.compact
    def __call__(self, x: Array, cond: Array) -> Array:
        out = nn.LayerNorm(epsilon=1e-5, name="norm")(x)
        film = nn.Dense(
            self.hidden_dim * 2,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="film",
        )(jax.nn.silu(cond))
        scale, bias = jnp.split(film, 2, axis=-1)
        out = out * (1.0 + scale) + bias
        out = nn.Dense(self.hidden_dim * self.expansion, name="expand")(out)
        out = jax.nn.silu(out)
        out = nn.Dense(
            self.hidden_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="project",
        )(out)
        return x + out


class ConditionalResidualMLP(nn.Module):
    """Conditional vector field for one-step MeanFlow.

    Observation and both MeanFlow times are encoded into one condition vector.
    Each residual block receives that condition through an independent FiLM
    projection. Intermediate block activations are returned for the existing
    dispersive regularizer.
    """

    input_dim: int
    global_cond_dim: int
    time_embed_dim: int
    hidden_dim: int
    num_res_blocks: int
    mlp_expansion: int

    @nn.compact
    def __call__(
        self,
        sample: Array,
        timestep: Array,
        r: Array,
        global_cond: Array,
    ) -> tuple[Array, tuple[Array, ...]]:
        t_embed = TimeEncoder(
            self.time_embed_dim, name="time_encoder"
        )(timestep)
        r_embed = TimeEncoder(
            self.time_embed_dim, name="start_time_encoder"
        )(r)
        cond = jnp.concatenate([global_cond, t_embed, r_embed], axis=-1)
        cond = nn.Dense(self.hidden_dim, name="condition_input")(cond)
        cond = jax.nn.silu(cond)
        cond = nn.Dense(self.hidden_dim, name="condition_output")(cond)

        x = nn.Dense(self.hidden_dim, name="sample_input")(sample)
        features: list[Array] = []
        for index in range(self.num_res_blocks):
            x = ConditionalResidualMLPBlock(
                hidden_dim=self.hidden_dim,
                expansion=self.mlp_expansion,
                name=f"residual_block_{index}",
            )(x, cond)
            # Two representative depths are sufficient for the O(B^2)
            # dispersive regularizer and avoid scaling its cost with depth.
            if index in (self.num_res_blocks // 2 - 1, self.num_res_blocks - 1):
                features.append(x)

        x = nn.LayerNorm(epsilon=1e-5, name="output_norm")(x)
        x = jax.nn.silu(x)
        velocity = nn.Dense(
            self.input_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="velocity_output",
        )(x)
        return velocity, tuple(features)



@jdc.pytree_dataclass
class Decoder1StepFMConfig:
    timestep_embed_dim: jdc.Static[int] = 128
    hidden_dim: jdc.Static[int] = 512
    num_res_blocks: jdc.Static[int] = 4
    mlp_expansion: jdc.Static[int] = 2
    condition_type: jdc.Static[str] = "film"

    policy_output_scale: float = 1.0
    learning_rate: float = 1e-4
    optimizer_beta1: float = 0.95
    optimizer_beta2: float = 0.999
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    batch_size: jdc.Static[int] = 128

    normalize_observations: jdc.Static[bool] = True
    normalization_mode: jdc.Static[str] = "gaussian"
    flow_ratio: float = 0.5
    time_dist: jdc.Static[str] = "lognorm"
    lognorm_mu: float = -0.4
    lognorm_sigma: float = 1.0
    adaptive_loss_gamma: float = 0.5
    adaptive_loss_c: float = 1e-3
    guidance_scale: float = 2.0
    use_dispersive: jdc.Static[bool] = False
    dispersive_loss_weight: float = 0.5
    bifm_loss_weight: float = 0.05
    warm_up_epoch = 0
    dispersive_tau: float = 1.0
    dispersive_chunk_size: jdc.Static[int] = 512
    use_lbifm: jdc.Static[bool] = False
    feather_std: float = 0.0
    latent_kl_weight: float = 1.0


@jdc.pytree_dataclass
class NormalizationStats:
    """Per-dimension statistics supporting MP1's [-1, 1] limits normalizer."""

    count: Array
    mean: Array
    var_sum: Array
    std: Array
    minimum: Array
    maximum: Array

    @staticmethod
    def init(shape: tuple[int, ...]) -> "NormalizationStats":
        return NormalizationStats(
            count=jnp.zeros(()),
            mean=jnp.zeros(shape),
            var_sum=jnp.zeros(shape),
            std=jnp.ones(shape),
            minimum=jnp.full(shape, jnp.inf),
            maximum=jnp.full(shape, -jnp.inf),
        )

    def update(self, x: Array) -> "NormalizationStats":
        axes = tuple(range(x.ndim - self.mean.ndim))
        batch_count = jnp.asarray(
            x.size // self.mean.size, dtype=self.count.dtype
        )
        batch_mean = jnp.mean(x, axis=axes)
        batch_var_sum = jnp.sum((x - batch_mean) ** 2, axis=axes)
        new_count = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / new_count
        new_var_sum = (
            self.var_sum
            + batch_var_sum
            + delta**2 * self.count * batch_count / new_count
        )
        variance = jnp.clip(new_var_sum / new_count, 1e-12, 1e12)
        return NormalizationStats(
            count=new_count,
            mean=new_mean,
            var_sum=new_var_sum,
            std=jnp.sqrt(variance),
            minimum=jnp.minimum(self.minimum, jnp.min(x, axis=axes)),
            maximum=jnp.maximum(self.maximum, jnp.max(x, axis=axes)),
        )


@jdc.pytree_dataclass
class Decoder1StepFMState:
    config: Decoder1StepFMConfig
    params: Any
    obs_stats: NormalizationStats
    obs_dim: jdc.Static[int]
    action_dim: jdc.Static[int]
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
    ) -> "Decoder1StepFMState":
        if config.condition_type != "film":
            raise ValueError("ConditionalResidualMLP supports FiLM conditioning.")
        if config.timestep_embed_dim < 4 or config.timestep_embed_dim % 2:
            raise ValueError("timestep_embed_dim must be even and at least 4.")
        if config.hidden_dim < 1 or config.num_res_blocks < 1:
            raise ValueError("hidden_dim and num_res_blocks must be positive.")
        if config.num_res_blocks % 2:
            raise ValueError("num_res_blocks must be even.")
        if config.mlp_expansion < 1:
            raise ValueError("mlp_expansion must be positive.")
        model = Decoder1StepFMState._make_model(config, obs_dim, action_dim)
        prng_params, prng_state = jax.random.split(prng)
        dummy_sample = jnp.zeros((1, action_dim))
        dummy_time = jnp.zeros((1,))
        dummy_obs = jnp.zeros((1, obs_dim))
        params = model.init(
            prng_params,
            dummy_sample,
            dummy_time,
            dummy_time,
            dummy_obs,
        )["params"]
        opt = Decoder1StepFMState._make_optimizer(config)
        return Decoder1StepFMState(
            config=config,
            params=params,
            obs_stats=NormalizationStats.init((obs_dim,)),
            obs_dim=obs_dim,
            action_dim=action_dim,
            opt=opt,
            opt_state=opt.init(params),
            prng=prng_state,
            steps=jnp.zeros((), dtype=jnp.int32),
        )

    @staticmethod
    def _make_optimizer(
        config: Decoder1StepFMConfig,
    ) -> optax.GradientTransformation:
        return optax.adamw(
            learning_rate=config.learning_rate,
            b1=config.optimizer_beta1,
            b2=config.optimizer_beta2,
            eps=config.optimizer_eps,
            weight_decay=config.optimizer_weight_decay,
        )

    @staticmethod
    def _make_model(
        config: Decoder1StepFMConfig, obs_dim: int, action_dim: int
    ) -> ConditionalResidualMLP:
        return ConditionalResidualMLP(
            input_dim=action_dim,
            global_cond_dim=obs_dim,
            time_embed_dim=config.timestep_embed_dim,
            hidden_dim=config.hidden_dim,
            num_res_blocks=config.num_res_blocks,
            mlp_expansion=config.mlp_expansion,
        )

    def _forward(
        self,
        params: Any,
        obs_norm: Array,
        x_t: Array,
        t: Array,
        r: Array,
    ) -> tuple[Array, tuple[Array, ...]]:
        model = self._make_model(self.config, self.obs_dim, self.action_dim)
        velocity, features = model.apply(
            {"params": params},
            x_t,
            t[:, 0],
            r[:, 0],
            obs_norm,
        )
        return velocity * self.config.policy_output_scale, features

    def meanflow_forward(
        self, obs_norm: Array, x_t: Array, t: Array, r: Array
    ) -> Array:
        velocity, _ = self._forward(self.params, obs_norm, x_t, t, r)
        return velocity

    def _normalize_obs(self, obs: Array) -> Array:
        if self.config.normalize_observations:
            return self._normalize_with_stats(obs, self.obs_stats)
        return obs

    def _normalize_with_stats(
        self, value: Array, stats: NormalizationStats
    ) -> Array:
        if self.config.normalization_mode == "gaussian":
            return (value - stats.mean) / (stats.std + 1e-8)
        if self.config.normalization_mode != "limits":
            raise ValueError(
                f"Unsupported normalization_mode: {self.config.normalization_mode}"
            )
        data_range = stats.maximum - stats.minimum
        regular = data_range >= 1e-4
        scale = jnp.where(regular, 2.0 / data_range, 1.0)
        offset = jnp.where(regular, -1.0 - scale * stats.minimum, -stats.minimum)
        return value * scale + offset

    def sample_action(
        self, obs: Array, prng: Array, deterministic: bool = False
    ) -> Array:
        obs_norm = self._normalize_obs(obs)
        single_obs = obs.ndim == 1
        if single_obs:
            obs_norm = obs_norm[None, :]
        batch_size = obs_norm.shape[0]
        prng_sample, prng_feather = jax.random.split(prng)
        z = jax.random.normal(prng_sample, (batch_size, self.action_dim))
        action = self._decode_normalized(obs_norm, z)
        if not deterministic:
            action += (
                jax.random.normal(prng_feather, action.shape)
                * self.config.feather_std
            )
        action = jnp.clip(action, -1.0, 1.0)
        return action[0] if single_obs else action

    def _decode_normalized(self, obs_norm: Array, z: Array) -> Array:
        t = jnp.ones((z.shape[0], 1))
        r = jnp.zeros((z.shape[0], 1))
        return z - self.meanflow_forward(obs_norm, z, t, r)

    def sample_action_from_z(
        self,
        obs: Array,
        z: Array,
        prng: Array,
        deterministic: bool = True,
    ) -> Array:
        obs_norm = self._normalize_obs(obs)
        single_obs = obs.ndim == 1
        if single_obs:
            obs_norm, z = obs_norm[None, :], z[None, :]
        action = self._decode_normalized(obs_norm, z)
        if not deterministic:
            action += jax.random.normal(prng, action.shape) * self.config.feather_std
        action = jnp.clip(action, -1.0, 1.0)
        return action[0] if single_obs else action

    def inverse_fm_batch(
        self,
        observations: Array,
        actions: Array,
    ) -> Array:
        """Invert actions with ``z <- action + u(obs, z, 1, 0)``.

        MeanFlow and the original FM decoder both operate directly in the
        environment action coordinate system; no action statistics are used.
        """
        if observations.ndim != 2 or actions.ndim != 2:
            raise ValueError(
                "MeanFlow inversion expects rank-2 observations and actions."
            )
        if observations.shape[0] != actions.shape[0]:
            raise ValueError(
                "MeanFlow inversion observations and actions must share a batch size."
            )
        return self._inverse_fm_batch_normalized(
            self._normalize_obs(observations), actions
        )

    def _inverse_fm_batch_normalized(
        self,
        obs_norm: Array,
        actions: Array,
        params: Any | None = None,
    ) -> Array:
        """Invert normalized actions using the supplied decoder parameters."""
        params = self.params if params is None else params
        t = jnp.zeros((actions.shape[0], 1))
        r = jnp.ones((actions.shape[0], 1))
        steps = 1
        return jax.lax.fori_loop(
            0,
            steps,
            lambda _, latent: actions
            + self._forward(params, obs_norm, latent, t, r)[0],
            actions,
        )

    def sample_t_r(
        self, prng: Array, batch_size: int
    ) -> tuple[Array, Array]:
        time_key, relation_key, less_key, greater_key = jax.random.split(prng, 4)
        # Sample the three (r, t) relations with equal probability:
        #   1/3: t ~ Uniform(0, 1), r ~ Uniform(0, t)
        #   1/3: t ~ Uniform(0, 1), r ~ Uniform(t, 1)
        #   1/3: t ~ Uniform(0, 1), r = t
        # Using one t per example keeps the requested conditional relations
        # exact while the independent relation draw provides the 1/3 mixture.
        t = jax.random.uniform(time_key, (batch_size,))
        relation = jax.random.uniform(relation_key, (batch_size,))
        r_less = t * jax.random.uniform(less_key, (batch_size,))
        r_greater = t + (1.0 - t) * jax.random.uniform(greater_key, (batch_size,))
        r = jnp.where(
            relation < 1.0 / 3.0,
            r_less,
            jnp.where(relation < 2.0 / 3.0, r_greater, t),
        )
        return t[:, None], r[:, None]

    def adaptive_l2_loss(
        self, error: Array, sample_mask: Array | None = None
    ) -> Array:
        delta_sq = jnp.mean(error**2, axis=-1)
        p = 1.0 - self.config.adaptive_loss_gamma
        weight = jax.lax.stop_gradient(
            1.0 / jnp.power(delta_sq + self.config.adaptive_loss_c, p)
        )
        if sample_mask is not None:
            sample_mask = sample_mask.astype(delta_sq.dtype)
            return jnp.sum(sample_mask * weight * delta_sq) / jnp.maximum(
                jnp.sum(sample_mask), 1.0
            )
        return jnp.mean(weight * delta_sq)

    def dispersive_loss(self, feature: Array) -> Array:
        """Compute the all-pairs loss without materializing a B x B x D array."""
        batch_size = feature.shape[0]
        chunk_size = min(self.config.dispersive_chunk_size, batch_size)
        padded_size = (
            (batch_size + chunk_size - 1) // chunk_size
        ) * chunk_size
        feature = jnp.pad(feature, ((0, padded_size - batch_size), (0, 0)))
        squared_norm = jnp.sum(feature**2, axis=-1)
        valid_columns = jnp.arange(padded_size) < batch_size
        num_chunks = padded_size // chunk_size

        def block_distance(block_index: int) -> tuple[Array, Array]:
            start = block_index * chunk_size
            block = jax.lax.dynamic_slice_in_dim(feature, start, chunk_size)
            block_norm = jax.lax.dynamic_slice_in_dim(
                squared_norm, start, chunk_size
            )
            distance = (
                block_norm[:, None]
                + squared_norm[None, :]
                - 2.0 * block @ feature.T
            )
            # Roundoff in the norm identity can produce tiny negative values.
            distance = jnp.maximum(distance, 0.0)
            valid_rows = (start + jnp.arange(chunk_size)) < batch_size
            valid = valid_rows[:, None] & valid_columns[None, :]
            return distance, valid

        def update_max(block_index: int, current_max: Array) -> Array:
            distance, valid = block_distance(block_index)
            return jnp.maximum(
                current_max,
                jnp.max(jnp.where(valid, distance, 0.0)),
            )

        max_distance = jax.lax.fori_loop(
            0, num_chunks, update_max, jnp.zeros((), dtype=feature.dtype)
        )
        scale = jnp.maximum(max_distance, 1e-8) * self.config.dispersive_tau

        def update_sum(block_index: int, current_sum: Array) -> Array:
            distance, valid = block_distance(block_index)
            values = jnp.exp(-distance / scale)
            return current_sum + jnp.sum(jnp.where(valid, values, 0.0))

        total = jax.lax.fori_loop(
            0, num_chunks, update_sum, jnp.zeros((), dtype=feature.dtype)
        )
        return jnp.log(total / (batch_size * batch_size))

    def compute_meanflow_loss(
        self,
        epoch,
        obs_norm: Array,
        action: Array,
        eps: Array,
        t: Array,
        r: Array,
        params: Any | None = None
    ) -> tuple[Array, Array, Array]:
        loss, meanflow_loss, dis_loss, _, _ = self._compute_training_losses(
            epoch, obs_norm, action, eps, t, r, params=params
        )
        return loss, meanflow_loss, dis_loss

    def compute_warm_up_bifm_weight(self, epoch):
        warm_up_epoch = self.config.warm_up_epoch
        # 使用 lax.cond 进行条件分支，两个分支都必须是函数
        weight = jax.lax.cond(
            epoch < warm_up_epoch,
            lambda: 0.0, 
            lambda: self.config.bifm_loss_weight                    
        )
        return weight
    
    def _compute_training_losses(
        self,
        epoch,
        obs_norm: Array,
        action: Array,
        eps: Array,
        t: Array,
        r: Array,
        params: Any | None = None,
    ) -> tuple[Array, Array, Array, Array, Array]:
        params = self.params if params is None else params
        x_t = t * eps + (1.0 - t) * action
        x_r = r * eps + (1.0 - r) * action
        v = eps - action

        def model_fn(
            z_arg: Array, t_arg: Array, r_arg: Array
        ) -> tuple[Array, tuple[Array, ...]]:
            return self._forward(params, obs_norm, z_arg, t_arg, r_arg)

        u_t, _ = model_fn(x_t, t, t)
        u_t = jax.lax.stop_gradient(u_t)
        v_hat = (
            self.config.guidance_scale * v
            + (1.0 - self.config.guidance_scale) * u_t
        )
        (u, features), (dudt, _) = jax.jvp(
            model_fn,
            (x_t, t, r),
            (v_hat, jnp.ones_like(t), jnp.zeros_like(r)),
        )
        target = jax.lax.stop_gradient(v_hat - (t - r) * dudt)
        meanflow_loss = self.adaptive_l2_loss(u - target)
        if self.config.use_dispersive:
            dis_loss = sum(
                (self.dispersive_loss(feature) for feature in features),
                start=jnp.zeros(()),
            )
        else:
            dis_loss = jnp.zeros(())
        inverse_latents = self._inverse_fm_batch_normalized(
            obs_norm, action, params=params
        )
        latent_mean = jnp.mean(inverse_latents, axis=0)
        latent_std = jnp.maximum(
            jnp.std(inverse_latents, axis=0), 1e-6
        )
        # Analytic KL[N(mu, sigma^2) || N(0, I)] for the empirical
        # distribution of latents obtained by inverting this action batch.
        prior_kl = 0.5 * jnp.sum(
            jnp.square(latent_mean)
            + jnp.square(latent_std)
            - 1.0
            - 2.0 * jnp.log(latent_std)
        )
        bifm_loss = jnp.zeros(())
        if self.config.use_lbifm:
            backward_u, _ = model_fn(x_r, r, t)
            nonzero_interval = jnp.squeeze(t != r, axis=-1)
            bifm_loss = self.adaptive_l2_loss(
                u + backward_u, sample_mask=nonzero_interval
            )
        loss = (
            meanflow_loss
            + self.config.dispersive_loss_weight * dis_loss
            + self.compute_warm_up_bifm_weight(epoch) * bifm_loss
            + self.config.latent_kl_weight * prior_kl
        )
        return loss, meanflow_loss, dis_loss, bifm_loss, prior_kl

    @jax.jit
    def train_step(
        self, epoch, batch_obs: Array, batch_actions: Array
    ) -> tuple["Decoder1StepFMState", dict[str, Array]]:
        batch_size = batch_obs.shape[0]
        obs_norm = self._normalize_obs(batch_obs)
        prng_eps, prng_tr, next_prng = jax.random.split(self.prng, 3)
        eps = jax.random.normal(prng_eps, batch_actions.shape)
        t, r = self.sample_t_r(prng_tr, batch_size)

        def loss_fn(params: Any):
            loss, meanflow_loss, dis_loss, bifm_loss, prior_kl = (
                self._compute_training_losses(
                    epoch, obs_norm, batch_actions, eps, t, r, params=params
                )
            )
            return loss, {
                "loss": loss,
                "meanflow_loss": meanflow_loss,
                "dis_loss": dis_loss,
                "bifm_loss": bifm_loss,
                "latent_prior_kl": prior_kl,
                "t_mean": jnp.mean(t),
                "r_mean": jnp.mean(r),
            }

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            self.params
        )
        updates, new_opt_state = self.opt.update(
            grads, self.opt_state, self.params
        )
        with jdc.copy_and_mutate(self) as new_state:
            new_state.params = optax.apply_updates(self.params, updates)
            new_state.opt_state = new_opt_state
            new_state.prng = next_prng
            new_state.steps = self.steps + 1
        return new_state, metrics
