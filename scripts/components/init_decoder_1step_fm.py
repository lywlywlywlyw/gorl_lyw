"""Create an identity one-step MeanFlow model for first iteration."""

import datetime
import pickle
from pathlib import Path
from typing import Annotated

import jax
import jax_dataclasses as jdc
import tyro
from jax import numpy as jnp
from mujoco_playground import dm_control_suite, locomotion, registry

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from flow_policy.decoder_1step_fm import Decoder1StepFMConfig, Decoder1StepFMState


def main(
    env_name: Annotated[
        str,
        tyro.conf.arg(
            constructor=tyro.extras.literal_type_from_choices(
                dm_control_suite.ALL_ENVS + locomotion.ALL_ENVS
            )
        ),
    ] = "CheetahRun",
    output_dir: str = "fm_models",
    seed: int = 42,
) -> None:
    """Create identity one-step MeanFlow checkpoint for first iteration."""

    env_config = registry.get_default_config(env_name)
    env = registry.load(env_name, config=env_config)

    obs_dim = env.observation_size
    action_dim = env.action_size

    config = Decoder1StepFMConfig(
        flow_steps=1,
        timestep_embed_dim=8,
        hidden_dims=(64, 64, 64, 64),
        policy_output_scale=1.0,
        learning_rate=3e-4,
        batch_size=2048,
        num_epochs=1,
        n_samples_per_action=8,
        normalize_observations=True,
        feather_std=0.0,
    )

    prng = jax.random.PRNGKey(seed)
    fm_state = Decoder1StepFMState.init(prng, obs_dim, action_dim, config)

    def zero_params(params):
        if isinstance(params, tuple):
            return tuple(zero_params(p) for p in params)
        if isinstance(params, list):
            return [zero_params(p) for p in params]
        return jnp.zeros_like(params)

    with jdc.copy_and_mutate(fm_state) as fm_state:
        fm_state.params = zero_params(fm_state.params)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = output_path / f"fm_1step_identity_{env_name}_{timestamp}.pkl"

    checkpoint = {
        "params": fm_state.params,
        "obs_stats": fm_state.obs_stats,
        "config": config,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "env_name": env_name,
        "is_identity": True,
        "is_1step_fm": True,
        "epoch": 0,
        "train_loss": 0.0,
        "val_loss": 0.0,
    }

    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"Init decoder (1-step FM): {checkpoint_file}")


if __name__ == "__main__":
    tyro.cli(main)
