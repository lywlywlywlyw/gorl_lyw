"""One-step MeanFlow decoder matching MP1's conditional 1D U-Net."""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax_dataclasses as jdc
import optax
from jax import Array
from jax import numpy as jnp

def _match_horizon(x: Array, target: int) -> Array:
    """Center-crop/pad NWC features to a skip connection's horizon."""
    current = x.shape[1]
    if current > target:
        start = (current - target) // 2
        return x[:, start : start + target, :]
    if current < target:
        total = target - current
        return jnp.pad(x, ((0, 0), (total // 2, total - total // 2), (0, 0)))
    return x


class SinusoidalPosEmb(nn.Module):
    """Exact sinusoidal basis used by MP1."""

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


class Conv1dBlock(nn.Module):
    out_channels: int
    kernel_size: int
    n_groups: int

    @nn.compact
    def __call__(self, x: Array) -> Array:
        x = nn.Conv(
            self.out_channels,
            kernel_size=(self.kernel_size,),
            padding="SAME",
        )(x)
        x = nn.GroupNorm(
            num_groups=self.n_groups,
            epsilon=1e-5,
        )(x)
        return jax.nn.mish(x)


class ConditionalResidualBlock1D(nn.Module):
    out_channels: int
    cond_dim: int
    kernel_size: int
    n_groups: int

    @nn.compact
    def __call__(self, x: Array, cond: Array) -> Array:
        residual = x
        out = Conv1dBlock(
            self.out_channels, self.kernel_size, self.n_groups
        )(x)

        film = nn.Dense(self.out_channels * 2)(jax.nn.mish(cond))
        scale, bias = jnp.split(film, 2, axis=-1)
        out = scale[:, None, :] * out + bias[:, None, :]

        out = Conv1dBlock(
            self.out_channels, self.kernel_size, self.n_groups
        )(out)
        if residual.shape[-1] != self.out_channels:
            residual = nn.Conv(
                self.out_channels, kernel_size=(1,), padding="SAME"
            )(residual)
        return out + residual


class ConditionalUnet1D(nn.Module):
    """JAX port of MP1 conditional_unet1d_meanflow_dis.py (FiLM path)."""

    input_dim: int
    global_cond_dim: int
    diffusion_step_embed_dim: int
    down_dims: tuple[int, ...]
    kernel_size: int
    n_groups: int
    use_down_condition: bool
    use_mid_condition: bool
    use_up_condition: bool

    @nn.compact
    def __call__(
        self,
        sample: Array,
        timestep: Array,
        r: Array,
        global_cond: Array,
    ) -> tuple[Array, tuple[Array, ...]]:
        original_horizon = sample.shape[1]
        t_embed = TimeEncoder(
            self.diffusion_step_embed_dim, name="diffusion_step_encoder"
        )(timestep)
        r_embed = TimeEncoder(
            self.diffusion_step_embed_dim, name="diffusion_step_encoder_rs"
        )(r)
        global_feature = jnp.concatenate(
            [t_embed + r_embed, global_cond], axis=-1
        )
        cond_dim = self.diffusion_step_embed_dim + self.global_cond_dim

        # Present in MP1's decoder parameterization, although its result is not
        # consumed by meanpolicy_dis.py.
        variance = sample
        for index in range(3):
            variance = nn.Dense(512, name=f"var_est_{index}")(variance)
            variance = jax.nn.silu(variance)
        _ = nn.Dense(1, name="var_est_3")(variance)

        all_dims = (self.input_dim,) + self.down_dims
        in_out = tuple(zip(all_dims[:-1], all_dims[1:]))
        x = sample
        skips: list[Array] = []
        down_latents: list[Array] = []

        for index, (_, dim_out) in enumerate(in_out):
            cond = global_feature if self.use_down_condition else jnp.zeros_like(
                global_feature
            )
            x = ConditionalResidualBlock1D(
                dim_out, cond_dim, self.kernel_size, self.n_groups,
                name=f"down_{index}_resnet_0",
            )(x, cond)
            x = ConditionalResidualBlock1D(
                dim_out, cond_dim, self.kernel_size, self.n_groups,
                name=f"down_{index}_resnet_1",
            )(x, cond)
            skips.append(x)
            down_latents.append(x.reshape((x.shape[0], -1)))
            if index < len(in_out) - 1:
                x = nn.Conv(
                    dim_out,
                    kernel_size=(3,),
                    strides=(2,),
                    padding="SAME",
                    name=f"down_{index}_sample",
                )(x)

        mid_dim = all_dims[-1]
        mid_cond = global_feature if self.use_mid_condition else jnp.zeros_like(
            global_feature
        )
        for index in range(2):
            x = ConditionalResidualBlock1D(
                mid_dim, cond_dim, self.kernel_size, self.n_groups,
                name=f"mid_{index}",
            )(x, mid_cond)

        reversed_pairs = tuple(reversed(in_out[1:]))
        for index, (dim_in, _) in enumerate(reversed_pairs):
            skip = skips.pop()
            x = _match_horizon(x, skip.shape[1])
            x = jnp.concatenate([x, skip], axis=-1)
            up_cond = (
                global_feature
                if self.use_up_condition
                else jnp.zeros_like(global_feature)
            )
            x = ConditionalResidualBlock1D(
                dim_in, cond_dim, self.kernel_size, self.n_groups,
                name=f"up_{index}_resnet_0",
            )(x, up_cond)
            x = ConditionalResidualBlock1D(
                dim_in, cond_dim, self.kernel_size, self.n_groups,
                name=f"up_{index}_resnet_1",
            )(x, up_cond)
            x = nn.ConvTranspose(
                dim_in,
                kernel_size=(4,),
                strides=(2,),
                padding="SAME",
                name=f"up_{index}_sample",
            )(x)

        x = _match_horizon(x, original_horizon)
        x = Conv1dBlock(
            self.down_dims[0], self.kernel_size, self.n_groups,
            name="final_block",
        )(x)
        velocity = nn.Conv(
            self.input_dim,
            kernel_size=(1,),
            padding="SAME",
            name="final_conv",
        )(x)
        return velocity, tuple(down_latents[:-1])

@jdc.pytree_dataclass
class Decoder1StepFMConfig:
    flow_steps: jdc.Static[int] = 1
    timestep_embed_dim: jdc.Static[int] = 256
    down_dims: jdc.Static[tuple[int, ...]] = (256, 512, 1024)
    kernel_size: jdc.Static[int] = 5
    n_groups: jdc.Static[int] = 8
    condition_type: jdc.Static[str] = "film"
    use_down_condition: jdc.Static[bool] = True
    use_mid_condition: jdc.Static[bool] = True
    use_up_condition: jdc.Static[bool] = True

    policy_output_scale: float = 1.0
    learning_rate: float = 1e-4
    optimizer_beta1: float = 0.95
    optimizer_beta2: float = 0.999
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    batch_size: jdc.Static[int] = 128
    num_epochs: jdc.Static[int] = 50
    # Retained only so older command lines/checkpoints remain readable.
    n_samples_per_action: jdc.Static[int] = 1

    normalize_observations: jdc.Static[bool] = True
    normalize_actions: jdc.Static[bool] = False
    normalization_mode: jdc.Static[str] = "limits"
    flow_ratio: float = 0.5
    time_dist: jdc.Static[str] = "lognorm"
    lognorm_mu: float = -0.4
    lognorm_sigma: float = 1.0
    adaptive_loss_gamma: float = 0.5
    adaptive_loss_c: float = 1e-3
    guidance_scale: float = 2.0
    dispersive_loss_weight: float = 0.5
    dispersive_tau: float = 1.0
    dispersive_chunk_size: jdc.Static[int] = 512
    use_lbifm: jdc.Static[bool] = False
    feather_std: float = 0.0


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
    action_stats: NormalizationStats
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
            raise ValueError("The MP1-compatible JAX decoder supports FiLM.")
        if len(config.down_dims) < 2:
            raise ValueError("ConditionalUnet1D requires at least two down_dims.")
        if config.timestep_embed_dim < 4 or config.timestep_embed_dim % 2:
            raise ValueError("timestep_embed_dim must be even and at least 4.")
        if any(dim % config.n_groups for dim in config.down_dims):
            raise ValueError("Every down_dim must be divisible by n_groups.")
        model = Decoder1StepFMState._make_model(config, obs_dim, action_dim)
        prng_params, prng_state = jax.random.split(prng)
        dummy_sample = jnp.zeros((1, 1, action_dim))
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
            action_stats=NormalizationStats.init((action_dim,)),
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
    ) -> ConditionalUnet1D:
        return ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim,
            diffusion_step_embed_dim=config.timestep_embed_dim,
            down_dims=config.down_dims,
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
            use_down_condition=config.use_down_condition,
            use_mid_condition=config.use_mid_condition,
            use_up_condition=config.use_up_condition,
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
            x_t[:, None, :],
            t[:, 0],
            r[:, 0],
            obs_norm,
        )
        return velocity[:, 0, :] * self.config.policy_output_scale, features

    def meanflow_forward(
        self, obs_norm: Array, x_t: Array, t: Array, r: Array
    ) -> Array:
        velocity, _ = self._forward(self.params, obs_norm, x_t, t, r)
        return velocity

    def _normalize_obs(self, obs: Array) -> Array:
        if self.config.normalize_observations:
            return self._normalize_with_stats(obs, self.obs_stats)
        return obs

    def _normalize_action(self, action: Array) -> Array:
        if self.config.normalize_actions:
            return self._normalize_with_stats(action, self.action_stats)
        return action

    def _unnormalize_action(self, action: Array) -> Array:
        if self.config.normalize_actions:
            stats = self.action_stats
            if self.config.normalization_mode == "gaussian":
                return action * (stats.std + 1e-8) + stats.mean
            data_range = stats.maximum - stats.minimum
            regular = data_range >= 1e-4
            scale = jnp.where(regular, 2.0 / data_range, 1.0)
            offset = jnp.where(regular, -1.0 - scale * stats.minimum, -stats.minimum)
            return (action - offset) / scale
        return action

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
        action = self._unnormalize_action(action)
        if not deterministic:
            action += (
                jax.random.normal(prng_feather, action.shape)
                * self.config.feather_std
            )
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
        action = self._unnormalize_action(self._decode_normalized(obs_norm, z))
        if not deterministic:
            action += jax.random.normal(prng, action.shape) * self.config.feather_std
        return action[0] if single_obs else action

    def sample_t_r(
        self, prng: Array, batch_size: int
    ) -> tuple[Array, Array]:
        prng_time, prng_flow = jax.random.split(prng)
        if self.config.time_dist == "uniform":
            samples = jax.random.uniform(prng_time, (batch_size, 2))
        elif self.config.time_dist == "lognorm":
            samples = jax.nn.sigmoid(
                jax.random.normal(prng_time, (batch_size, 2))
                * self.config.lognorm_sigma
                + self.config.lognorm_mu
            )
        else:
            raise ValueError(f"Unsupported time_dist: {self.config.time_dist}")
        t = jnp.maximum(samples[:, 0], samples[:, 1])
        r = jnp.minimum(samples[:, 0], samples[:, 1])
        mask = (
            jax.random.uniform(prng_flow, (batch_size,))
            < self.config.flow_ratio
        )
        r = jnp.where(mask, t, r)
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
        obs_norm: Array,
        action: Array,
        eps: Array,
        t: Array,
        r: Array,
        params: Any | None = None,
    ) -> tuple[Array, Array, Array]:
        loss, meanflow_loss, dis_loss, _ = self._compute_training_losses(
            obs_norm, action, eps, t, r, params=params
        )
        return loss, meanflow_loss, dis_loss

    def _compute_training_losses(
        self,
        obs_norm: Array,
        action: Array,
        eps: Array,
        t: Array,
        r: Array,
        params: Any | None = None,
    ) -> tuple[Array, Array, Array, Array]:
        params = self.params if params is None else params
        action_norm = self._normalize_action(action)
        x_t = t * eps + (1.0 - t) * action_norm
        x_r = r * eps + (1.0 - r) * action_norm
        v = eps - action_norm

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
        dis_loss = sum(
            (self.dispersive_loss(feature) for feature in features),
            start=jnp.zeros(()),
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
            + 2*bifm_loss
        )
        return loss, meanflow_loss, dis_loss, bifm_loss

    @jax.jit
    def train_step(
        self, batch_obs: Array, batch_actions: Array
    ) -> tuple["Decoder1StepFMState", dict[str, Array]]:
        batch_size = batch_obs.shape[0]
        obs_norm = self._normalize_obs(batch_obs)
        prng_eps, prng_tr, next_prng = jax.random.split(self.prng, 3)
        eps = jax.random.normal(prng_eps, batch_actions.shape)
        t, r = self.sample_t_r(prng_tr, batch_size)

        def loss_fn(params: Any):
            loss, meanflow_loss, dis_loss, bifm_loss = (
                self._compute_training_losses(
                    obs_norm, batch_actions, eps, t, r, params=params
                )
            )
            return loss, {
                "loss": loss,
                "meanflow_loss": meanflow_loss,
                "dis_loss": dis_loss,
                "bifm_loss": bifm_loss,
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
