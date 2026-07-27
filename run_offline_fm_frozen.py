"""Two-stage offline FM training: converge decoder, freeze it, train IQL encoder.

All dataset formats and loading semantics are inherited from ``run_offline_fm``.
Enable Weights & Biases with ``--wandb-mode online`` (disabled by default).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax_dataclasses as jdc
import numpy as np
import optax
import tyro
from jax import Array
from jax import numpy as jnp
from tqdm import trange

import run_offline_fm as base
from flow_policy import networks
from datetime import datetime

PyTree = Any


@dataclass
class FrozenDecoderConfig(base.OfflineConfig):
    """Configuration for the fixed-decoder two-stage method."""

    # Match the reference offline run: a no-argument launch resolves this D4RL
    # task through base._download_d4rl_dataset instead of looking for the old
    # local-only datasets/data.pkl default. Both fields remain CLI-overridable.
    data_path: str | None = None
    d4rl_dataset: str | None = "walker2d-medium-expert-v2"
    output_dir: str = "results/offline_fm_frozen_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    # Decoder convergence and held-out evaluation.
    decoder_max_epochs: int = 200
    decoder_min_epochs: int = 20
    decoder_patience: int = 20
    decoder_validation_fraction: float = 0.05
    decoder_min_delta: float = 1e-4
    decoder_eval_batches: int = 32

    # Once the decoder is frozen, this is the total number of IQL updates.
    encoder_iql_steps: int = 500_000
    comparison_samples: int = 4096

    # W&B is opt-in so offline/local runs remain dependency-service independent.
    wandb_project: str = "offline-fm"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_name: str | None = None
    wandb_mode: str = "disabled"


class WandbLogger:
    """Small optional W&B adapter with an identical no-op interface."""

    def __init__(self, config: FrozenDecoderConfig, method: str):
        self.run = None
        if config.wandb_mode == "disabled":
            return
        try:
            import wandb
        except ImportError as error:
            raise ImportError(
                "W&B logging requires `wandb`; install requirements.txt or "
                "run with --wandb-mode disabled."
            ) from error
        self.run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            group=config.wandb_group,
            name=config.wandb_name,
            mode=config.wandb_mode,
            config={**asdict(config), "method": method},
            tags=[method],
        )

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def validate_config(config: FrozenDecoderConfig) -> None:
    if config.decoder_max_epochs < 1:
        raise ValueError("decoder_max_epochs must be positive.")
    if not 1 <= config.decoder_min_epochs <= config.decoder_max_epochs:
        raise ValueError("decoder_min_epochs must be in [1, decoder_max_epochs].")
    if config.decoder_patience < 1:
        raise ValueError("decoder_patience must be positive.")
    if not 0.0 < config.decoder_validation_fraction < 1.0:
        raise ValueError("decoder_validation_fraction must be between 0 and 1.")
    if config.encoder_iql_steps < 1 or config.comparison_samples < 1:
        raise ValueError("encoder_iql_steps and comparison_samples must be positive.")
    if config.batch_size < 1 or config.decoder_batch_size < 1:
        raise ValueError("Batch sizes must be positive.")
    if config.latent_inverse_steps < 1:
        raise ValueError("latent_inverse_steps must be positive.")
    if config.wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline, or disabled.")


def split_indices(
    size: int, validation_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if size < 2:
        raise ValueError("At least two transitions are required for validation.")
    permutation = rng.permutation(size)
    validation_size = min(size - 1, max(1, int(size * validation_fraction)))
    return permutation[validation_size:], permutation[:validation_size]


def decoder_validation_loss(
    decoder: base.DecoderFMState,
    buffer: base.ReplayBuffer,
    indices: np.ndarray,
    max_batches: int,
    key: Array,
) -> tuple[float, Array]:
    """Evaluate held-out CFM without changing decoder state."""
    losses: list[float] = []
    batch_size = decoder.config.batch_size
    for batch_number, start in enumerate(range(0, len(indices), batch_size)):
        if batch_number >= max_batches:
            break
        batch = indices[start : start + batch_size]
        obs = jnp.asarray(buffer.observations[batch])
        actions = jnp.asarray(buffer.actions[batch])
        obs_norm = (obs - decoder.obs_stats.mean) / (decoder.obs_stats.std + 1e-8)
        key, eps_key, time_key = jax.random.split(key, 3)
        eps = jax.random.normal(
            eps_key,
            (len(batch), decoder.config.n_samples_per_action, actions.shape[-1]),
        )
        times = jax.random.uniform(
            time_key, (len(batch), decoder.config.n_samples_per_action, 1)
        )
        losses.append(
            float(jnp.mean(decoder.compute_cfm_loss(obs_norm, actions, eps, times)))
        )
    return float(np.mean(losses)), key


def forward_fm_batch(
    decoder: base.DecoderFMState, observations: Array, latents: Array
) -> Array:
    """Deterministic generation with no random allocation in the metric path."""
    obs_norm = (observations - decoder.obs_stats.mean) / (
        decoder.obs_stats.std + 1e-8
    )

    def step(x_t: Array, pair: tuple[Array, Array]) -> tuple[Array, None]:
        t_current, t_next = pair
        t = jnp.full((*x_t.shape[:-1], 1), t_current)
        velocity = decoder.flow_forward(obs_norm, x_t, decoder.embed_timestep(t))
        return x_t + (t_next - t_current) * velocity, None

    schedule = decoder.get_schedule()
    actions, _ = jax.lax.scan(
        step, latents, (schedule.t_current, schedule.t_next)
    )
    return actions


def decoder_comparison_metrics(
    decoder: base.DecoderFMState,
    buffer: base.ReplayBuffer,
    indices: np.ndarray,
    inverse_steps: int,
) -> tuple[dict[str, float], np.ndarray]:
    """Metrics that test whether inverse latents and actions are usable."""
    obs = jnp.asarray(buffer.observations[indices])
    actions = jnp.asarray(buffer.actions[indices])
    latents = jax.jit(
        lambda current_obs, current_actions: base.inverse_fm_batch(
            decoder, current_obs, current_actions, inverse_steps
        )
    )(obs, actions)
    reconstructed = jax.jit(
        lambda current_obs, current_latents: forward_fm_batch(
            decoder, current_obs, current_latents
        )
    )(obs, latents)
    latent_mean = jnp.mean(latents, axis=0)
    latent_std = jnp.std(latents, axis=0)
    metrics = {
        "decoder/cycle_action_mse": float(jnp.mean((reconstructed - actions) ** 2)),
        "decoder/cycle_action_mae": float(jnp.mean(jnp.abs(reconstructed - actions))),
        "latent/mean_abs": float(jnp.mean(jnp.abs(latent_mean))),
        "latent/std_mean": float(jnp.mean(latent_std)),
        "latent/std_error": float(jnp.mean(jnp.abs(latent_std - 1.0))),
        "latent/norm_mean": float(jnp.mean(jnp.linalg.norm(latents, axis=-1))),
        "latent/max_abs": float(jnp.max(jnp.abs(latents))),
    }
    return metrics, np.asarray(latents)


def policy_comparison_metrics(
    actor_params: PyTree,
    q1_params: PyTree,
    q2_params: PyTree,
    value_params: PyTree,
    decoder: base.DecoderFMState,
    buffer: base.ReplayBuffer,
    normalized_observations: np.ndarray,
    indices: np.ndarray,
    latent_targets: np.ndarray,
) -> dict[str, float]:
    """Common offline policy proxies for comparing both training methods."""
    obs_norm = jnp.asarray(normalized_observations[indices])
    obs_raw = jnp.asarray(buffer.observations[indices])
    data_actions = jnp.asarray(buffer.actions[indices])
    target_z = jnp.asarray(latent_targets[indices])
    distribution = networks.gaussian_policy_fwd(actor_params, obs_norm)
    policy_z = distribution.loc
    policy_actions = jax.jit(
        lambda current_obs, current_latents: forward_fm_batch(
            decoder, current_obs, current_latents
        )
    )(obs_raw, policy_z)
    q_policy = jnp.minimum(
        networks.q_mlp_fwd(q1_params, obs_norm, policy_actions),
        networks.q_mlp_fwd(q2_params, obs_norm, policy_actions),
    )
    q_data = jnp.minimum(
        networks.q_mlp_fwd(q1_params, obs_norm, data_actions),
        networks.q_mlp_fwd(q2_params, obs_norm, data_actions),
    )
    value = networks.value_mlp_fwd(value_params, obs_norm)
    latent_nll = -jnp.mean(
        jnp.sum(distribution.log_prob(target_z), axis=-1)
    )
    return {
        "comparison/policy_q": float(jnp.mean(q_policy)),
        "comparison/data_q": float(jnp.mean(q_data)),
        "comparison/policy_q_minus_data_q": float(jnp.mean(q_policy - q_data)),
        "comparison/policy_q_minus_v": float(jnp.mean(q_policy - value)),
        "comparison/policy_action_data_mse": float(
            jnp.mean((policy_actions - data_actions) ** 2)
        ),
        "comparison/policy_action_abs_mean": float(jnp.mean(jnp.abs(policy_actions))),
        "comparison/policy_action_outside_unit_fraction": float(
            jnp.mean(jnp.abs(policy_actions) > 1.0)
        ),
        "encoder/latent_nll": float(latent_nll),
        "encoder/mean_target_mse": float(jnp.mean((policy_z - target_z) ** 2)),
        "encoder/scale_mean": float(jnp.mean(distribution.scale)),
    }


def initialize_training(
    config: FrozenDecoderConfig, buffer: base.ReplayBuffer
) -> tuple[Any, ...]:
    """Initialize the shared decoder, actor and IQL networks."""
    obs_dim = buffer.observations.shape[-1]
    action_dim = buffer.actions.shape[-1]
    key = jax.random.key(config.seed)
    key, decoder_key, actor_key, q1_key, q2_key, value_key = jax.random.split(
        key, 6
    )
    obs_mean = jnp.asarray(buffer.observations.mean(axis=0))
    obs_std = jnp.asarray(buffer.observations.std(axis=0) + 1e-6)
    observations = (buffer.observations - np.asarray(obs_mean)) / np.asarray(obs_std)
    next_observations = (
        buffer.next_observations - np.asarray(obs_mean)
    ) / np.asarray(obs_std)

    decoder = base.init_decoder(config, obs_dim, action_dim, decoder_key)
    with jdc.copy_and_mutate(decoder) as decoder:
        decoder.obs_stats = decoder.obs_stats.init((obs_dim,)).update(
            jnp.asarray(buffer.observations)
        )

    hidden_dims = (config.hidden_size,) * config.hidden_layers
    actor_params = networks.mlp_init(
        actor_key, (obs_dim, *hidden_dims, action_dim * 2)
    )
    q1_params = networks.mlp_init(
        q1_key, (obs_dim + action_dim, *hidden_dims, 1)
    )
    q2_params = networks.mlp_init(
        q2_key, (obs_dim + action_dim, *hidden_dims, 1)
    )
    value_params = networks.mlp_init(value_key, (obs_dim, *hidden_dims, 1))
    return (
        key, decoder, actor_params, q1_params, q2_params, value_params,
        obs_mean, obs_std, observations, next_observations,
    )


def make_optimizers(
    config: FrozenDecoderConfig,
    actor_params: PyTree,
    q1_params: PyTree,
    q2_params: PyTree,
    value_params: PyTree,
) -> tuple[Any, ...]:
    actor_optimizer = optax.adam(config.actor_learning_rate)
    critic_optimizer = optax.adam(config.critic_learning_rate)
    value_optimizer = optax.adam(config.value_learning_rate)
    return (
        actor_optimizer,
        critic_optimizer,
        value_optimizer,
        actor_optimizer.init(actor_params),
        critic_optimizer.init((q1_params, q2_params)),
        value_optimizer.init(value_params),
        base.make_iql_update(
            config, actor_optimizer, critic_optimizer, value_optimizer
        ),
    )


def run_iql(
    *,
    config: FrozenDecoderConfig,
    method: str,
    logger: WandbLogger,
    metrics_file: Path,
    global_step: int,
    num_steps: int,
    rng: np.random.Generator,
    buffer: base.ReplayBuffer,
    normalized_observations: np.ndarray,
    normalized_next_observations: np.ndarray,
    latent_targets: np.ndarray,
    decoder: base.DecoderFMState,
    actor_params: PyTree,
    actor_opt_state: PyTree,
    q1_params: PyTree,
    q2_params: PyTree,
    critic_opt_state: PyTree,
    value_params: PyTree,
    value_opt_state: PyTree,
    target_q1_params: PyTree,
    target_q2_params: PyTree,
    iql_update: Any,
) -> tuple[Any, ...]:
    accumulators: dict[str, list[float]] = {}
    comparison_indices = rng.choice(
        len(buffer), min(config.comparison_samples, len(buffer)), replace=False
    )
    for step in trange(num_steps, desc="IQL", leave=False):
        indices = rng.integers(0, len(buffer), size=config.batch_size)
        (
            actor_params, actor_opt_state, q1_params, q2_params,
            critic_opt_state, value_params, value_opt_state,
            target_q1_params, target_q2_params, metrics,
        ) = iql_update(
            actor_params, actor_opt_state, q1_params, q2_params,
            critic_opt_state, value_params, value_opt_state,
            target_q1_params, target_q2_params,
            jnp.asarray(normalized_observations[indices]),
            jnp.asarray(buffer.actions[indices]),
            jnp.asarray(buffer.rewards[indices]),
            jnp.asarray(normalized_next_observations[indices]),
            jnp.asarray(buffer.masks[indices]),
            jnp.asarray(latent_targets[indices]),
        )
        for name, value in metrics.items():
            accumulators.setdefault(name, []).append(float(value))
        if (step + 1) % config.log_interval == 0 or step + 1 == num_steps:
            record = {
                "method": method,
                "phase": "encoder",
                "global_step": global_step + step + 1,
                "encoder/iql_step": step + 1,
                **{
                    f"iql/{name}": float(np.mean(values))
                    for name, values in accumulators.items()
                },
                **policy_comparison_metrics(
                    actor_params, q1_params, q2_params, value_params, decoder,
                    buffer, normalized_observations, comparison_indices,
                    latent_targets,
                ),
            }
            logger.log(
                {k: v for k, v in record.items() if isinstance(v, (int, float))},
                global_step + step + 1,
            )
            with open(metrics_file, "a") as file:
                file.write(json.dumps(record) + "\n")
            print(f"  step {step + 1}: {record}")
            accumulators.clear()
    return (
        actor_params, actor_opt_state, q1_params, q2_params,
        critic_opt_state, value_params, value_opt_state,
        target_q1_params, target_q2_params, global_step + num_steps,
    )


def main(config: FrozenDecoderConfig) -> None:
    validate_config(config)
    method = "frozen_decoder"
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as file:
        json.dump({**asdict(config), "method": method}, file, indent=2)
    metrics_file = output_dir / "metrics.jsonl"
    logger = WandbLogger(config, method)
    rng = np.random.default_rng(config.seed)

    try:
        buffer = base.load_replay_buffer(config)
        (
            key, decoder, actor_params, q1_params, q2_params, value_params,
            obs_mean, obs_std, observations, next_observations,
        ) = initialize_training(config, buffer)
        train_indices, validation_indices = split_indices(
            len(buffer), config.decoder_validation_fraction, rng
        )
        comparison_indices = rng.choice(
            validation_indices,
            min(config.comparison_samples, len(validation_indices)),
            replace=False,
        )
        print(
            f"Loaded {len(buffer):,} transitions. Training decoder once, then "
            "freezing it permanently."
        )

        best_params = decoder.params
        best_validation = float("inf")
        stale_epochs = 0
        global_step = 0
        previous_validation_latents: np.ndarray | None = None
        for epoch in trange(config.decoder_max_epochs, desc="Decoder epochs"):
            epoch_losses: list[float] = []
            permutation = rng.permutation(train_indices)
            for start in range(0, len(permutation), config.decoder_batch_size):
                batch = permutation[start : start + config.decoder_batch_size]
                decoder, metrics = decoder.train_step(
                    jnp.asarray(buffer.observations[batch]),
                    jnp.asarray(buffer.actions[batch]),
                )
                epoch_losses.append(float(metrics["loss"]))
                global_step += 1
            validation_loss, key = decoder_validation_loss(
                decoder, buffer, validation_indices,
                config.decoder_eval_batches, key,
            )
            comparison, validation_latents = decoder_comparison_metrics(
                decoder, buffer, comparison_indices, config.latent_inverse_steps
            )
            comparison["latent/drift_mse"] = (
                0.0
                if previous_validation_latents is None
                else float(
                    np.mean(
                        (validation_latents - previous_validation_latents) ** 2
                    )
                )
            )
            previous_validation_latents = validation_latents
            record = {
                "method": method,
                "phase": "decoder",
                "global_step": global_step,
                "decoder/epoch": epoch + 1,
                "decoder/train_cfm_loss": float(np.mean(epoch_losses)),
                "decoder/validation_cfm_loss": validation_loss,
                **comparison,
            }
            logger.log(
                {k: v for k, v in record.items() if isinstance(v, (int, float))},
                global_step,
            )
            with open(metrics_file, "a") as file:
                file.write(json.dumps(record) + "\n")

            improved = validation_loss < best_validation - config.decoder_min_delta
            if improved:
                best_validation = validation_loss
                best_params = jax.tree.map(jnp.copy, decoder.params)
                stale_epochs = 0
            else:
                stale_epochs += 1
            if (
                epoch + 1 >= config.decoder_min_epochs
                and stale_epochs >= config.decoder_patience
            ):
                print(f"Decoder early-stopped at epoch {epoch + 1}.")
                break

        with jdc.copy_and_mutate(decoder) as decoder:
            decoder.params = best_params
        print(f"Frozen decoder validation CFM loss: {best_validation:.6f}")

        latent_targets = base.build_latent_targets(
            decoder, buffer, config.decoder_batch_size,
            config.latent_inverse_steps,
        )
        final_decoder_metrics, _ = decoder_comparison_metrics(
            decoder, buffer, comparison_indices, config.latent_inverse_steps
        )
        frozen_record = {
            "method": method,
            "phase": "decoder_frozen",
            "global_step": global_step,
            "decoder/best_validation_cfm_loss": best_validation,
            **final_decoder_metrics,
        }
        logger.log(
            {
                k: v for k, v in frozen_record.items()
                if isinstance(v, (int, float))
            },
            global_step,
        )
        with open(metrics_file, "a") as file:
            file.write(json.dumps(frozen_record) + "\n")

        (
            actor_optimizer, critic_optimizer, value_optimizer,
            actor_opt_state, critic_opt_state, value_opt_state, iql_update,
        ) = make_optimizers(
            config, actor_params, q1_params, q2_params, value_params
        )
        del actor_optimizer, critic_optimizer, value_optimizer
        target_q1_params = jax.tree.map(jnp.copy, q1_params)
        target_q2_params = jax.tree.map(jnp.copy, q2_params)
        (
            actor_params, actor_opt_state, q1_params, q2_params,
            critic_opt_state, value_params, value_opt_state,
            target_q1_params, target_q2_params, global_step,
        ) = run_iql(
            config=config, method=method, logger=logger,
            metrics_file=metrics_file, global_step=global_step,
            num_steps=config.encoder_iql_steps, rng=rng, buffer=buffer,
            normalized_observations=observations,
            normalized_next_observations=next_observations,
            latent_targets=latent_targets, decoder=decoder,
            actor_params=actor_params, actor_opt_state=actor_opt_state,
            q1_params=q1_params, q2_params=q2_params,
            critic_opt_state=critic_opt_state, value_params=value_params,
            value_opt_state=value_opt_state,
            target_q1_params=target_q1_params,
            target_q2_params=target_q2_params, iql_update=iql_update,
        )
        del (
            actor_opt_state, critic_opt_state, value_opt_state,
            target_q1_params, target_q2_params, global_step,
        )
        base.save_checkpoint(
            output_dir / "checkpoint_final.pkl", config, decoder, actor_params,
            q1_params, q2_params, value_params, obs_mean, obs_std, 0,
        )
        print(f"Saved final checkpoint to {output_dir / 'checkpoint_final.pkl'}")
    finally:
        logger.finish()


if __name__ == "__main__":
    main(tyro.cli(FrozenDecoderConfig))
