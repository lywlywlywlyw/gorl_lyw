"""Collect data from a latent-space encoder + FM for training a new FM."""

import datetime
import pickle
import time
import json
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
from flow_policy import encoder_ppo, encoder_rlpd
from flow_policy.decoder_fm import DecoderFMState
from flow_policy.agent import EncoderFMAgent
from flow_policy.rollout_encoder import (
    BatchedRolloutStateEncoderFM,
    eval_policy_encoder_fm
)
from envs.robomimic.RobomimicEnv import RobomimicEnv
from envs.robomimic.online_config.training_config import TrainingConfig
from envs.robomimic.online_config.env_config import EnvConfig
try:
    from .metrics_ipc import append_metrics
    from .online_pipeline_ipc import ChunkReplayBuffer, VersionManager
except ImportError:  # Direct execution: python scripts/components/collect_data_fm.py
    from metrics_ipc import append_metrics
    from online_pipeline_ipc import ChunkReplayBuffer, VersionManager


def _load_policy_pair(
    encoder_path: Path,
    decoder_path: Path,
    env: RobomimicEnv,
    config: dict,
) -> tuple[EncoderFMAgent, bool]:
    """Load one explicit matching policy pair; never reads trainer memory."""
    with encoder_path.open("rb") as file:
        encoder_checkpoint = pickle.load(file)
    with decoder_path.open("rb") as file:
        decoder_checkpoint = pickle.load(file)
    # Offline combined checkpoints use ``config`` for the FM decoder and keep
    # the encoder config separately. Online encoder checkpoints use ``config``.
    encoder_config = encoder_checkpoint.get(
        "rlpd_encoder_config", encoder_checkpoint["config"]
    )
    encoder_state = encoder_rlpd.EncoderState.init(
        jax.random.key(config["seed"]), env, encoder_config
    )
    with jdc.copy_and_mutate(encoder_state) as state:
        state.actor_params = encoder_checkpoint["rlpd_z_actor_params"]
        state.critic_params = encoder_checkpoint["rlpd_z_critic_params"]
        state.target_critic_params = encoder_checkpoint["rlpd_z_target_critic_params"]
        state.log_temperature = encoder_checkpoint["rlpd_z_log_temperature"]
        state.obs_stats = encoder_checkpoint["rlpd_z_obs_stats"]
        for name in ("actor_opt_state", "critic_opt_state", "temperature_opt_state", "prng", "steps"):
            key = f"rlpd_z_{name}"
            if key in encoder_checkpoint:
                setattr(state, name, encoder_checkpoint[key])
    decoder_state = DecoderFMState.init(
        jax.random.PRNGKey(config["seed"] + 1000),
        decoder_checkpoint["obs_dim"], decoder_checkpoint["action_dim"],
        decoder_checkpoint["config"],
    )
    with jdc.copy_and_mutate(decoder_state) as state:
        state.params = decoder_checkpoint["params"]
        state.obs_stats = decoder_checkpoint["obs_stats"]
    return EncoderFMAgent(ppo_z_state=encoder_state, fm_state=decoder_state), bool(
        encoder_config.apply_tanh_in_rollout
    )


def run_async_collector(
    pipeline_root: str,
    replay_buffer_dir: str,
    stop_file: str,
    poll_seconds: float = 2.0,
    rollout_steps: int = 100,
    replay_capacity: int | None = None,
    metrics_file: str | None = None,
) -> None:
    """Continuously collect real transitions with the latest complete Policy_n."""
    config = TrainingConfig().to_dict() | EnvConfig().to_dict()
    manager = VersionManager(pipeline_root)
    replay = ChunkReplayBuffer(replay_buffer_dir, replay_capacity)
    stop = Path(stop_file)
    env = RobomimicEnv(
        dataset_path=config["dataset_path"], reward_shaping=config["dense_reward"]
    )
    rollout_state = BatchedRolloutStateEncoderFM.init(
        env, jax.random.key(config["seed"] + 1), config["num_envs"]
    )
    current_version = -1
    agent = None
    apply_tanh = True
    try:
        while not stop.exists():
            latest = manager.latest_policy()
            if latest is None:
                time.sleep(poll_seconds)
                continue
            version, encoder_path, decoder_path = latest
            if version != current_version:
                agent, apply_tanh = _load_policy_pair(
                    encoder_path, decoder_path, env, config
                )
                current_version = version
                if metrics_file:
                    append_metrics(metrics_file, {
                        "collector/policy_version": version,
                        "collector/policy_switch": 1,
                    })
            assert agent is not None
            rollout_state, transitions = rollout_state.rollout(
                agent,
                episode_length=config["episode_length"],
                iterations_per_env=rollout_steps,
                apply_tanh_in_rollout=apply_tanh,
            )
            rewards = onp.asarray(jax.device_get(transitions.reward)).reshape(-1)
            discounts = onp.asarray(jax.device_get(transitions.discount)).reshape(-1)
            truncations = onp.asarray(jax.device_get(transitions.truncation)).reshape(-1).astype(bool)
            payload = {
                "observations": onp.asarray(jax.device_get(transitions.obs)).reshape(-1, int(env.observation_size)),
                "actions": onp.asarray(jax.device_get(transitions.action_info.env_action)).reshape(-1, int(env.action_size)),
                "rewards": rewards,
                "next_observations": onp.asarray(jax.device_get(transitions.next_obs)).reshape(-1, int(env.observation_size)),
                "masks": discounts,
                "dones": onp.logical_and(discounts == 0.0, ~truncations),
                "truncations": truncations,
            }
            replay.append(payload, metadata={
                "policy_version": version,
                "encoder_checkpoint": str(encoder_path),
                "decoder_checkpoint": str(decoder_path),
            })
            if metrics_file:
                append_metrics(metrics_file, {
                    "collector/policy_version": version,
                    "collector/transitions": len(rewards),
                    "collector/replay_size": replay.size(),
                    "collector/reward_mean": float(rewards.mean()),
                })
    finally:
        rollout_state.close()

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

    # Load encoder checkpoint. The legacy CLI argument name is retained so the
    # existing PPO pipeline remains backward compatible.
    with open(ppo_z_checkpoint_path, "rb") as f:
        encoder_checkpoint = pickle.load(f)

    if "rlpd_z_actor_params" in encoder_checkpoint:
        encoder_algorithm = "rlpd"
    elif "ppo_z_params" in encoder_checkpoint:
        encoder_algorithm = "ppo"
    else:
        raise ValueError(
            "Encoder checkpoint is neither RLPD nor PPO; expected "
            "'rlpd_z_actor_params' or 'ppo_z_params'."
        )

    # Load FM model config (needed for initialization)
    with open(fm_model_path, "rb") as f:
        fm_config_source = pickle.load(f)

    # Setup environment
    env = RobomimicEnv(dataset_path=config['dataset_path'], reward_shaping=config['dense_reward'])
    z_dim = env.action_size 
    # Get config from checkpoint or create new one
    if "config" in encoder_checkpoint:
        encoder_config = encoder_checkpoint["config"]
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

    # Reconstruct the algorithm-specific encoder state.
    if encoder_algorithm == "rlpd":
        ppo_z_state = encoder_rlpd.EncoderState.init(
            prng=jax.random.key(config['seed']),
            env=env,
            config=encoder_config,
        )
        with jdc.copy_and_mutate(ppo_z_state) as ppo_z_state:
            ppo_z_state.actor_params = encoder_checkpoint["rlpd_z_actor_params"]
            ppo_z_state.critic_params = encoder_checkpoint["rlpd_z_critic_params"]
            ppo_z_state.target_critic_params = encoder_checkpoint[
                "rlpd_z_target_critic_params"
            ]
            ppo_z_state.log_temperature = encoder_checkpoint[
                "rlpd_z_log_temperature"
            ]
            ppo_z_state.obs_stats = encoder_checkpoint["rlpd_z_obs_stats"]
            for state_name in (
                "actor_opt_state",
                "critic_opt_state",
                "temperature_opt_state",
                "prng",
                "steps",
            ):
                checkpoint_key = f"rlpd_z_{state_name}"
                if checkpoint_key in encoder_checkpoint:
                    setattr(ppo_z_state, state_name, encoder_checkpoint[checkpoint_key])
        apply_tanh_in_rollout = encoder_config.apply_tanh_in_rollout
    else:
        ppo_z_state = encoder_ppo.EncoderState.init(
            prng=jax.random.key(config['seed']),
            env=env,
            config=encoder_config
        )
        with jdc.copy_and_mutate(ppo_z_state) as ppo_z_state:
            ppo_z_state.params = encoder_checkpoint["ppo_z_params"]
            ppo_z_state.obs_stats = encoder_checkpoint["ppo_z_obs_stats"]
        apply_tanh_in_rollout = config['ppo_apply_tanh_in_rollout']

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
        if "fm_params" in encoder_checkpoint and "fm_obs_stats" in encoder_checkpoint:
            fm_state.params = encoder_checkpoint["fm_params"]
            fm_state.obs_stats = encoder_checkpoint["fm_obs_stats"]
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
        apply_tanh_in_rollout=apply_tanh_in_rollout,
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
            apply_tanh_in_rollout=apply_tanh_in_rollout,
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
    data_file = output_path / (
        f"{encoder_algorithm}_z_fm_data_{config['env_name']}_{timestamp}.pkl"
    )

    data = {
        "states": all_states,
        "actions": all_actions,
        "rewards": all_rewards,
        "env_name": config['env_name'],
        "config": encoder_config,
        "collection_method": f"{encoder_algorithm}_z_fm_rollout",
        "encoder_algorithm": encoder_algorithm,
        "encoder_checkpoint": ppo_z_checkpoint_path,
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