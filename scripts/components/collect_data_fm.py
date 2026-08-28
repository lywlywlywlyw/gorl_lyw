"""Collect data from a latent-space encoder + FM for training a new FM."""

import datetime
import pickle
import time
import json
import warnings
from pathlib import Path
from typing import Annotated

import jax
import jax_dataclasses as jdc
import numpy as onp
import cv2
import tyro
from jax import numpy as jnp
# from mujoco_playground import dm_control_suite, locomotion, registry
# from mujoco_playground.config import dm_control_suite_params
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from flow_policy import encoder_ppo, encoder_rlpd
from flow_policy.decoder_fm import DecoderFMState
from flow_policy.decoder_1step_fm_residualMLP import Decoder1StepFMState
from flow_policy.agent import EncoderFMAgent
from flow_policy.rollout_encoder import (
    BatchedRolloutStateEncoderFM,
    eval_policy_encoder_fm
)
from envs.base_env import State
from flow_policy.rollout_encoder import SUCCESS_REWARD_BONUS
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
    with jdc.copy_and_mutate(encoder_state) as encoder_state:
        encoder_state.actor_params = encoder_checkpoint["rlpd_z_actor_params"]
        encoder_state.critic_params = encoder_checkpoint["rlpd_z_critic_params"]
        encoder_state.target_critic_params = encoder_checkpoint["rlpd_z_target_critic_params"]
        encoder_state.log_temperature = encoder_checkpoint["rlpd_z_log_temperature"]
        encoder_state.obs_stats = encoder_checkpoint["rlpd_z_obs_stats"]
        for name in ("actor_opt_state", "critic_opt_state", "temperature_opt_state", "prng", "steps"):
            key = f"rlpd_z_{name}"
            if key in encoder_checkpoint:
                setattr(encoder_state, name, encoder_checkpoint[key])
    decoder_type = decoder_checkpoint.get("decoder_type", config["decoder_type"])
    if decoder_type != config["decoder_type"]: raise ValueError("Policy checkpoint decoder type does not match pipeline type.")
    state_cls = Decoder1StepFMState if decoder_type == "meanflow" else DecoderFMState
    decoder_state = state_cls.init(jax.random.PRNGKey(config["seed"] + 1000), decoder_checkpoint["obs_dim"], decoder_checkpoint["action_dim"], decoder_checkpoint["config"])
    with jdc.copy_and_mutate(decoder_state) as decoder_state:
        decoder_state.params, decoder_state.obs_stats = decoder_checkpoint["params"], decoder_checkpoint["obs_stats"]
    return EncoderFMAgent(ppo_z_state=encoder_state, fm_state=decoder_state), bool(
        encoder_config.apply_tanh_in_rollout
    )


def _record_policy_evaluation(
    agent: EncoderFMAgent,
    config: dict,
    version: int,
    output_path: Path | None,
    apply_tanh_in_rollout: bool,
) -> dict[str, float | int | str]:
    """Evaluate a newly published policy and record the first episode."""
    num_episodes = int(config["eval_num_envs"])
    if num_episodes < 1:
        raise ValueError("eval_num_envs must be positive.")
    video_env = RobomimicEnv(
        dataset_path=config["dataset_path"],
        render_offscreen=True,
        reward_shaping=config["dense_reward"],
    )
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    writer = None
    try:
        for episode in range(num_episodes):
            key = jax.random.key(config["seed"] + version * 10000 + episode)
            state = video_env.reset(key)
            episode_return = 0.0
            success = False
            length = 0
            if episode == 0 and output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    max(1, int(config["wandb_video_fps"])),
                    (int(config["wandb_video_width"]), int(config["wandb_video_height"])),
                )
                if not writer.isOpened():
                    writer.release()
                    writer = None
                    raise RuntimeError(f"Could not create evaluation video: {output_path}")
            for step in range(int(config["episode_length"])):
                if writer is not None and step % max(1, int(config["wandb_video_frame_skip"])) == 0:
                    frame = video_env.render(
                        mode="rgb_array",
                        width=int(config["wandb_video_width"]),
                        height=int(config["wandb_video_height"]),
                    )
                    writer.write(cv2.cvtColor(onp.asarray(frame, dtype=onp.uint8), cv2.COLOR_RGB2BGR))
                key, sample_key = jax.random.split(key)
                obs = jnp.expand_dims(state.obs, axis=0)
                latent, _ = agent.sample_z(obs, sample_key, deterministic=True)
                action = agent.map_z_to_action(obs, latent)[0]
                if apply_tanh_in_rollout:
                    action = jnp.tanh(action)
                state = video_env.step(state, action)
                episode_return += float(onp.asarray(state.reward))
                length = step + 1
                for success_key in ("success", "task_success", "is_success"):
                    if success_key in state.info:
                        value = state.info[success_key]
                        if isinstance(value, dict):
                            value = value.get("task", False)
                        success = success or bool(onp.asarray(value))
                if bool(onp.asarray(state.done)):
                    break
            if writer is not None:
                frame = video_env.render(
                    mode="rgb_array",
                    width=int(config["wandb_video_width"]),
                    height=int(config["wandb_video_height"]),
                )
                writer.write(cv2.cvtColor(onp.asarray(frame, dtype=onp.uint8), cv2.COLOR_RGB2BGR))
                writer.release()
                writer = None
            returns.append(episode_return)
            lengths.append(length)
            successes.append(float(success))
    finally:
        if writer is not None:
            writer.release()
        video_env.close()
    metrics: dict[str, float | int | str] = {
        "pipeline/version": version,
        "eval/return_mean": float(onp.mean(returns)),
        "eval/return_std": float(onp.std(returns)),
        "eval/return_min": float(onp.min(returns)),
        "eval/return_max": float(onp.max(returns)),
        "eval/episode_length_mean": float(onp.mean(lengths)),
        "eval/success_rate": float(onp.mean(successes)),
    }
    if output_path is not None:
        metrics["_video_path"] = str(output_path.resolve())
    return metrics


def _record_q_gap_evaluation(
    agent: EncoderFMAgent,
    config: dict,
    version: int,
    replay_buffer_dir: str,
    warn: bool = True,
) -> dict[str, float | int]:
    """Compare current-critic estimates with restored-state MC returns.

    The first action in each Monte Carlo rollout must be generated from the
    exact same latent used for the critic estimate.  Also, bind the decoder
    from this explicitly loaded evaluation agent rather than relying on an
    agent-level mapping that could accidentally refer to another decoder.
    """
    replay = ChunkReplayBuffer(replay_buffer_dir)
    try:
        data = replay.load_snapshot()
    except (FileNotFoundError, ValueError):
        if warn:
            warnings.warn("Skipping Q-gap evaluation because replay is empty.")
        return {}

    states = data.get("env_states")
    if states is None:
        if warn:
            warnings.warn("Skipping Q-gap evaluation because replay has no env_states.")
        return {}
    valid = [
        index
        for index, state in enumerate(states)
        if isinstance(state, dict)
        and "qpos" in state
        and "qvel" in state
        and "time" in state
    ]
    if not valid:
        if warn:
            warnings.warn(
                "Skipping Q-gap evaluation because replay has no valid MuJoCo env_states."
            )
        return {}

    invalid_count = len(states) - len(valid)
    if invalid_count and warn:
        warnings.warn(
            f"Skipping {invalid_count} replay entries with missing or invalid env_states."
        )

    sample_count = min(30, len(valid))
    rng = onp.random.default_rng(int(config["seed"]) + version)
    indices = rng.choice(valid, size=sample_count, replace=False)
    eval_env = RobomimicEnv(
        dataset_path=config["dataset_path"],
        reward_shaping=config["dense_reward"],
    )
    estimated_values: list[float] = []
    true_values: list[float] = []
    gamma = float(config["rlpd_discounting"])
    # ``agent`` is loaded by the evaluator from the matching
    # (encoder_version, decoder_version) pair.  Keep this decoder fixed for
    # the whole evaluation, so Q-gap uses Decoder_n rather than any trainer
    # state or a subsequently updated decoder.
    evaluation_decoder = agent.fm_state
    try:
        for index in indices:
            eval_env.set_env_state(states[index])
            observation = jnp.asarray(data["observations"][index], dtype=jnp.float32)[None, :]
            latent, _ = agent.sample_z(
                observation,
                jax.random.key(int(config["seed"]) + version * 1000 + int(index)),
                deterministic=True,
            )
            q_values = agent.ppo_z_state._critic_values(
                agent.ppo_z_state.critic_params,
                observation,
                latent,
            )
            estimated_values.append(float(onp.asarray(jnp.mean(q_values))))

            state = State(
                obs=observation[0],
                reward=jnp.asarray(0.0),
                done=jnp.asarray(False),
                info={},
            )
            discounted_return = 0.0
            discount = 1.0
            for step in range(int(config["episode_length"])):
                if step == 0:
                    # Reuse the latent used by the critic above.  Do not
                    # sample a second, independent z for the first return
                    # step, even if the policy is stochastic in the future.
                    return_latent = latent
                else:
                    return_latent, _ = agent.sample_z(
                        state.obs[None, :],
                        jax.random.key(
                            int(config["seed"])
                            + version * 100000
                            + int(index) * 1000
                            + step
                        ),
                        deterministic=True,
                    )
                action = evaluation_decoder.sample_action_from_z(
                    state.obs[None, :],
                    return_latent,
                    jax.random.PRNGKey(0),
                    deterministic=True,
                )[0]
                state = eval_env.step(state, action)
                reward = float(onp.asarray(state.reward))
                success = eval_env.is_success()
                if success:
                    reward += SUCCESS_REWARD_BONUS
                discounted_return += discount * reward
                discount *= gamma
                if bool(onp.asarray(state.done)) or success:
                    break
            true_values.append(discounted_return)
    finally:
        eval_env.close()

    estimated = float(onp.mean(estimated_values))
    true = float(onp.mean(true_values))
    return {
        "pipeline/version": version,
        "estimated_value": estimated,
        "true_value": true,
        "q_gap": estimated - true,
        "q_gap/num_samples": sample_count,
    }


def run_async_collector(
    pipeline_root: str,
    replay_buffer_dir: str,
    stop_file: str,
    poll_seconds: float = 2.0,
    rollout_steps: int = 100,
    replay_capacity: int | None = None,
    metrics_file: str | None = None,
    decoder_type: str = "flow_matching",
) -> None:
    """Continuously collect transitions with the latest complete Policy_n.

    Policy evaluation is handled by run_async_evaluator, so slow evaluation
    cannot block collection or skip intermediate policy versions.
    """
    config = TrainingConfig().to_dict() | EnvConfig().to_dict()
    config["decoder_type"] = decoder_type
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
                        "pipeline/version": version,
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
                "dones": (discounts == 0.0),
                "truncations": truncations,
                "env_states": onp.asarray(
                    rollout_state.last_transition_env_states, dtype=object
                ),
            }
            replay.append(payload, metadata={
                "policy_version": version,
                "encoder_checkpoint": str(encoder_path),
                "decoder_checkpoint": str(decoder_path),
            })
            if metrics_file:
                append_metrics(metrics_file, {
                    "pipeline/version": version,
                    "collector/policy_version": version,
                    "collector/transitions": len(rewards),
                    "collector/replay_size": replay.size(),
                    "collector/reward_mean": float(rewards.mean()),
                })
    finally:
        rollout_state.close()


def run_async_evaluator(
    pipeline_root: str,
    stop_file: str,
    max_version: int,
    poll_seconds: float = 2.0,
    metrics_file: str | None = None,
    evaluation_dir: str | None = None,
    replay_buffer_dir: str | None = None,
    decoder_type: str = "flow_matching",
) -> None:
    """Evaluate every Policy_n exactly once and strictly in version order."""
    if max_version < 0:
        raise ValueError("max_version must be non-negative")

    config = TrainingConfig().to_dict() | EnvConfig().to_dict()
    config["decoder_type"] = decoder_type
    manager = VersionManager(pipeline_root)
    stop = Path(stop_file)
    env = RobomimicEnv(
        dataset_path=config["dataset_path"], reward_shaping=config["dense_reward"]
    )
    video_interval = max(1, int(config["wandb_video_interval_evals"]))
    try:
        # Waiting for an explicit version, rather than latest_policy(), ensures
        # that a quickly advancing trainer cannot skip evaluations.
        for version in range(max_version + 1):
            encoder_path = manager.wait_component(
                "encoder", version, poll_seconds, stop
            )
            decoder_path = manager.wait_component(
                "decoder", version, poll_seconds, stop
            )
            agent, apply_tanh = _load_policy_pair(
                encoder_path, decoder_path, env, config
            )
            record_video = (version + 1) % video_interval == 0
            metrics = _record_policy_evaluation(
                agent,
                config,
                version,
                (
                    Path(evaluation_dir) / f"policy_{version:04d}.mp4"
                    if record_video and evaluation_dir is not None
                    else None
                ),
                apply_tanh,
            )
            if replay_buffer_dir is not None:
                # The evaluator starts concurrently with the collector.  In
                # particular, Policy_0 is the offline checkpoint and its q-gap
                # must wait for the first restored simulator states instead of
                # being permanently skipped because replay is still empty.
                q_gap_deadline = time.monotonic() + 300.0
                while not stop.exists():
                    q_gap = _record_q_gap_evaluation(
                        agent, config, version, replay_buffer_dir, warn=False
                    )
                    if q_gap:
                        metrics.update(q_gap)
                        break
                    if time.monotonic() >= q_gap_deadline:
                        metrics.update(_record_q_gap_evaluation(
                            agent, config, version, replay_buffer_dir, warn=True
                        ))
                        break
                    time.sleep(poll_seconds)
            if metrics_file:
                append_metrics(metrics_file, metrics)
            print(f"Evaluation completed for Policy_{version}.", flush=True)
    finally:
        env.close()


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