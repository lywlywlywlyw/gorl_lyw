"""Collect data from PPO_z + FM for training new FM."""

import datetime
import pickle
import time
from pathlib import Path
from typing import Annotated

import jax
import jax_dataclasses as jdc
import numpy as onp
import tyro
from jax import numpy as jnp
# from mujoco_playground import dm_control_suite, locomotion, registry
# from mujoco_playground.config import dm_control_suite_params
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from flow_policy import encoder_ppo
from flow_policy.decoder_fm import DecoderFMState
from flow_policy.agent import EncoderFMAgent
from flow_policy.rollout_encoder import (
    BatchedRolloutStateEncoderFM,
    eval_policy_encoder_fm
)
from envs.robomimic.RobomimicEnv import RobomimicEnv
from envs.robomimic.online_config.training_config import TrainingConfig
from envs.robomimic.online_config.env_config import EnvConfig
def main(
    ppo_z_checkpoint_path: str | None = None,
    fm_model_path: str | None = None,
    output_dir: str = "data"
) -> None:
    config = TrainingConfig().to_dict() | EnvConfig().to_dict()
    """Collect data from PPO_z + FM combined policy."""

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Auto-detect checkpoints if not provided
    if ppo_z_checkpoint_path is None:
        # Find the latest best checkpoint from PPO_z training for this specific environment
        import glob
        checkpoints = glob.glob(f"results/ppo_z_fm_v2_{config['env_name']}_*/best_checkpoint.pkl")
        if checkpoints:
            ppo_z_checkpoint_path = sorted(checkpoints)[-1]
        else:
            raise ValueError(f"No PPO_z checkpoint found for {config['env_name']}. Please specify --ppo_z_checkpoint_path")

    if fm_model_path is None:
        # Find the latest FM model
        import glob
        fm_models = glob.glob("fm_models/fm_model_best_*.pkl")
        if fm_models:
            fm_model_path = sorted(fm_models)[-1]
        else:
            raise ValueError("No FM model found. Please specify --fm_model_path")

    # Load PPO_z checkpoint
    with open(ppo_z_checkpoint_path, "rb") as f:
        ppo_z_checkpoint = pickle.load(f)

    # Load FM model config (needed for initialization)
    with open(fm_model_path, "rb") as f:
        fm_config_source = pickle.load(f)

    # Setup environment
    env = RobomimicEnv(dataset_path=config['dataset_path'])
    z_dim = env.action_size 
    # Get config from checkpoint or create new one
    if "config" in ppo_z_checkpoint:
        encoder_config = ppo_z_checkpoint["config"]
    else:
        # Create config with z_dim
        encoder_config = encoder_ppo.EncoderConfig(action_repeat=config['action_repeat'],
        batch_size=config['ppo_batch_size'],
        discounting=config['ppo_discounting'],
        entropy_cost=config['ppo_entropy_cost'],
        episode_length=config['episode_length'],
        learning_rate=config['ppo_learning_rate'],
        normalize_observations=config['ppo_normalize_observations'],
        num_envs=config['num_envs'],
        num_evals=config['ppo_num_evals'],
        num_minibatches=config['ppo_num_minibatches'],
        num_timesteps=config['ppo_num_timesteps'],
        num_updates_per_batch=config['ppo_num_updates_per_batch'],
        reward_scaling=config['ppo_reward_scaling'],
        unroll_length=config['ppo_unroll_length'],
        z_dim=z_dim,
        gae_lambda=config['ppo_gae_lambda'],
        normalize_advantage=config['ppo_normalize_advantage'],
        clipping_epsilon=config['ppo_clipping_epsilon'],
        value_loss_coeff=config['ppo_value_loss_coeff'],
        z_regularization=config['ppo_z_regularization'],
        max_grad_norm=config['ppo_max_grad_norm'],
        use_tanh_jacobian_for_z=config['ppo_use_tanh_jacobian_for_z'],)

    # Initialize PPO_z state
    ppo_z_state = encoder_ppo.EncoderState.init(
        prng=jax.random.key(config['seed']),
        env=env,
        config=encoder_config
    )

    # Load PPO_z parameters
    with jdc.copy_and_mutate(ppo_z_state) as ppo_z_state:
        ppo_z_state.params = ppo_z_checkpoint["ppo_z_params"]
        ppo_z_state.obs_stats = ppo_z_checkpoint["ppo_z_obs_stats"]

    # Initialize FM state
    fm_prng = jax.random.PRNGKey(config['seed'] + 1000)
    fm_state = DecoderFMState.init(
        fm_prng,
        fm_config_source['obs_dim'],
        fm_config_source['action_dim'],
        fm_config_source['config']
    )

    # Load FM parameters from PPO_z checkpoint (not from standalone FM file)
    # This ensures we use the exact FM that was trained with PPO_z
    with jdc.copy_and_mutate(fm_state) as fm_state:
        if "fm_params" in ppo_z_checkpoint and "fm_obs_stats" in ppo_z_checkpoint:
            fm_state.params = ppo_z_checkpoint["fm_params"]
            fm_state.obs_stats = ppo_z_checkpoint["fm_obs_stats"]
        else:
            # Fallback: use standalone FM (shouldn't happen but safe)
            fm_state.params = fm_config_source["params"]
            fm_state.obs_stats = fm_config_source["obs_stats"]

    # Create combined agent
    agent = EncoderFMAgent(
        ppo_z_state=ppo_z_state,
        fm_state=fm_state,
    )

    # Initialize rollout state
    rollout_state = BatchedRolloutStateEncoderFM.init(
        env,
        prng=jax.random.key(config['seed'] + 1),
        num_envs=config['num_envs'],
    )

    # Validate first
    eval_outputs = eval_policy_encoder_fm(
        agent,
        prng=jax.random.fold_in(agent.ppo_z_state.prng, 0),
        num_envs=config['eval_num_envs'],
        max_episode_length=config['episode_length'],
    )
    s_np = {k: onp.array(v) for k, v in eval_outputs.scalar_metrics.items()}

    # Collect data
    all_states = []
    all_actions = []
    all_rewards = []
    config['ppo_iterations_per_env'] = (config['ppo_num_minibatches'] * config['ppo_batch_size'] * config['ppo_unroll_length']) // config['num_envs']
    for i in tqdm(range(config['data_collection_iterations']), desc="Collecting"):
        # Custom rollout that saves actual actions (not z values)
        rollout_state, states, actions, rewards = rollout_state.rollout_with_actions(
            agent,
            episode_length=config['episode_length'],
            iterations_per_env=config['ppo_iterations_per_env'],
        )

        all_states.append(onp.array(states))
        all_actions.append(onp.array(actions))
        all_rewards.append(onp.array(rewards))

    # Combine all data
    all_states = onp.concatenate(all_states, axis=0)
    all_actions = onp.concatenate(all_actions, axis=0)
    all_rewards = onp.concatenate(all_rewards, axis=0)

    # Reshape to (num_samples, dim)
    T, B = all_states.shape[:2]
    all_states = all_states.reshape(-1, all_states.shape[-1])
    all_actions = all_actions.reshape(-1, all_actions.shape[-1])
    all_rewards = all_rewards.reshape(-1)


    # Save data
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = output_path / f"ppo_z_fm_data_{config['env_name']}_{timestamp}.pkl"

    data = {
        "states": all_states,
        "actions": all_actions,
        "rewards": all_rewards,
        "env_name": config['env_name'],
        "config": encoder_config,
        "collection_method": "ppo_z_fm_rollout",
        "ppo_z_checkpoint": ppo_z_checkpoint_path,
        "fm_model": fm_model_path,
        "num_iterations": config['data_collection_iterations'],
        "total_samples": len(all_states),
        "expected_episode_reward": s_np['reward_mean'],
    }

    with open(data_file, "wb") as f:
        pickle.dump(data, f)

    print(f"Collect data: {len(all_states)} samples -> {data_file}")


if __name__ == "__main__":
    tyro.cli(main)