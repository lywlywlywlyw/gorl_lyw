"""Alternating offline FM decoder refinement and latent-space IQL training.

This preserves the original alternating algorithm while adding held-out CFM,
inverse/cycle, latent, IQL and offline policy-comparison metrics. Dataset
loading is inherited unchanged from ``run_offline_fm``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import numpy as np
import tyro
from jax import numpy as jnp
from tqdm import trange
from datetime import datetime
import run_offline_fm as base
from run_offline_fm_frozen import (
    FrozenDecoderConfig,
    WandbLogger,
    decoder_comparison_metrics,
    decoder_validation_loss,
    initialize_training,
    make_optimizers,
    run_iql,
    split_indices,
    validate_config,
)


@dataclass
class AlternatingConfig(FrozenDecoderConfig):
    """Configuration for the original alternating method."""

    # Keep this entry point self-contained as well as inheriting the frozen
    # defaults, so its no-argument behavior is explicit and stable.
    data_path: str | None = None
    d4rl_dataset: str | None = "walker2d-medium-expert-v2"
    output_dir: str = "results/offline_fm_alternating_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def main(config: AlternatingConfig) -> None:
    validate_config(config)
    if config.num_rounds < 1 or config.decoder_epochs_per_round < 1:
        raise ValueError("num_rounds and decoder_epochs_per_round must be positive.")
    if config.iql_steps_per_round < 1:
        raise ValueError("iql_steps_per_round must be positive.")

    method = "alternating"
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
        (
            actor_optimizer, critic_optimizer, value_optimizer,
            actor_opt_state, critic_opt_state, value_opt_state, iql_update,
        ) = make_optimizers(
            config, actor_params, q1_params, q2_params, value_params
        )
        del actor_optimizer, critic_optimizer, value_optimizer
        target_q1_params = jax.tree.map(jnp.copy, q1_params)
        target_q2_params = jax.tree.map(jnp.copy, q2_params)
        global_step = 0
        previous_validation_latents: np.ndarray | None = None

        print(
            f"Loaded {len(buffer):,} transitions. Running "
            f"{config.num_rounds} alternating rounds."
        )
        for round_index in range(config.num_rounds):
            print(f"\nRound {round_index + 1}/{config.num_rounds}: decoder")
            for epoch in trange(
                config.decoder_epochs_per_round,
                desc="Decoder epochs",
                leave=False,
            ):
                losses: list[float] = []
                permutation = rng.permutation(train_indices)
                for start in range(0, len(permutation), config.decoder_batch_size):
                    batch = permutation[start : start + config.decoder_batch_size]
                    decoder, metrics = decoder.train_step(
                        jnp.asarray(buffer.observations[batch]),
                        jnp.asarray(buffer.actions[batch]),
                    )
                    losses.append(float(metrics["loss"]))
                    global_step += 1
                validation_loss, key = decoder_validation_loss(
                    decoder, buffer, validation_indices,
                    config.decoder_eval_batches, key,
                )
                comparison, validation_latents = decoder_comparison_metrics(
                    decoder, buffer, comparison_indices,
                    config.latent_inverse_steps,
                )
                comparison["latent/drift_mse"] = (
                    0.0
                    if previous_validation_latents is None
                    else float(
                        np.mean(
                            (
                                validation_latents
                                - previous_validation_latents
                            )
                            ** 2
                        )
                    )
                )
                previous_validation_latents = validation_latents
                record = {
                    "method": method,
                    "phase": "decoder",
                    "round": round_index,
                    "global_step": global_step,
                    "decoder/epoch": (
                        round_index * config.decoder_epochs_per_round + epoch + 1
                    ),
                    "decoder/train_cfm_loss": float(np.mean(losses)),
                    "decoder/validation_cfm_loss": validation_loss,
                    **comparison,
                }
                logger.log(
                    {
                        k: v for k, v in record.items()
                        if isinstance(v, (int, float))
                    },
                    global_step,
                )
                with open(metrics_file, "a") as file:
                    file.write(json.dumps(record) + "\n")

            latent_targets = base.build_latent_targets(
                decoder, buffer, config.decoder_batch_size,
                config.latent_inverse_steps,
            )
            print(f"Round {round_index + 1}/{config.num_rounds}: encoder + IQL")
            (
                actor_params, actor_opt_state, q1_params, q2_params,
                critic_opt_state, value_params, value_opt_state,
                target_q1_params, target_q2_params, global_step,
            ) = run_iql(
                config=config, method=method, logger=logger,
                metrics_file=metrics_file, global_step=global_step,
                num_steps=config.iql_steps_per_round, rng=rng, buffer=buffer,
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
            base.save_checkpoint(
                output_dir / f"checkpoint_round_{round_index:03d}.pkl",
                config, decoder, actor_params, q1_params, q2_params,
                value_params, obs_mean, obs_std, round_index,
            )

        base.save_checkpoint(
            output_dir / "checkpoint_final.pkl", config, decoder, actor_params,
            q1_params, q2_params, value_params, obs_mean, obs_std,
            config.num_rounds - 1,
        )
        print(f"Saved final checkpoint to {output_dir / 'checkpoint_final.pkl'}")
    finally:
        logger.finish()


if __name__ == "__main__":
    main(tyro.cli(AlternatingConfig))
