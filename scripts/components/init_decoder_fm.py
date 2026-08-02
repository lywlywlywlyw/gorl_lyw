"""Create an identity Flow Matching model for first iteration.

This creates an FM checkpoint where z = a (identity mapping).
The velocity field is initialized to zero, so flow matching doesn't change the input.
"""

import datetime
import pickle
from pathlib import Path
from typing import Annotated

import jax
import tyro
from jax import numpy as jnp
# from mujoco_playground import dm_control_suite, locomotion, registry

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from flow_policy.decoder_fm import DecoderFMConfig, DecoderFMState
from envs.robomimic.RobomimicEnv import RobomimicEnv
from envs.robomimic.config.decoder_configs.fm_config import FlowMatchingConfig

def main(
    env_name: str = "Lift",
    dataset_path: str = "/root/GoRL/datasets/robomimic/low_dim.hdf5",
    output_dir: str = "fm_models",
    seed: int = 42,
) -> None:
    """Create identity FM checkpoint for first iteration.

    Args:
        env_name: Environment name (to determine obs_dim and action_dim)
        output_dir: Directory to save the checkpoint
        seed: Random seed for initialization
    """
    fm_configs = FlowMatchingConfig().to_dict()
    # Load environment to get dimensions
    env = RobomimicEnv(dataset_path=dataset_path)

    obs_dim = env.observation_size
    action_dim = env.action_size

    # Create FM config
    # Use minimal network since we want identity mapping
    config = DecoderFMConfig(
        flow_steps=fm_configs["fm_flow_steps"],
        timestep_embed_dim=fm_configs["fm_timestep_embed_dim"],
        hidden_dims=fm_configs["fm_hidden_dims"],
        policy_output_scale=fm_configs["fm_policy_output_scale"],
        learning_rate=fm_configs["fm_learning_rate"],
        batch_size=fm_configs["fm_batch_size"],
        num_epochs=fm_configs["fm_num_epochs"],  # Not used for identity
        n_samples_per_action=fm_configs["fm_n_samples_per_action"],
        normalize_observations=fm_configs["fm_normalize_observations"],
        sde_sigma=fm_configs["fm_sde_sigma"],
        feather_std=fm_configs["fm_feather_std"],
    )

    # Initialize FM state
    prng = jax.random.PRNGKey(seed)
    fm_state = DecoderFMState.init(prng, obs_dim, action_dim, config)

    # Zero out all network parameters to create identity mapping
    # When velocity = 0, x_t stays constant during integration
    # So if we start from z, we end at z (identity mapping)

    def zero_params(params):
        """Recursively zero all parameters."""
        if isinstance(params, tuple):
            return tuple(zero_params(p) for p in params)
        elif isinstance(params, list):
            return [zero_params(p) for p in params]
        else:
            # It's a JAX array
            return jnp.zeros_like(params)

    zeroed_params = zero_params(fm_state.params)

    # Update FM state with zeroed parameters
    import jax_dataclasses as jdc
    with jdc.copy_and_mutate(fm_state) as fm_state:
        fm_state.params = zeroed_params

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = output_path / f"fm_identity_{env_name}_{timestamp}.pkl"

    checkpoint = {
        "params": fm_state.params,
        "obs_stats": fm_state.obs_stats,
        "config": config,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "env_name": env_name,
        "is_identity": True,  # Mark this as identity checkpoint
        "epoch": 0,
        "train_loss": 0.0,
        "val_loss": 0.0,
    }

    with open(checkpoint_file, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"Init decoder (FM): {checkpoint_file}")


if __name__ == "__main__":
    tyro.cli(main)
