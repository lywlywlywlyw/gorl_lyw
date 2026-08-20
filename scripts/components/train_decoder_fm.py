"""Train Flow Matching model to learn PPO action distribution."""

import datetime
import pickle
from pathlib import Path

import jax
import jax_dataclasses as jdc
import numpy as np
import tyro
from jax import numpy as jnp
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from flow_policy.decoder_fm import DecoderFMConfig, DecoderFMState
from envs.robomimic.online_config.training_config import TrainingConfig
from envs.robomimic.online_config.env_config import EnvConfig
from metrics_ipc import append_metrics

def train_fm(
    data_path: str = "data/ppo_training_data_WalkerWalk_20250928_212057.pkl",
    output_dir: str = "fm_models",
    stage: int = 0,
    global_epoch_offset: int = 0,
    wandb_run_id: str | None = None,
    wandb_run_name: str | None = None,
    metrics_file: str | None = None,
    stage_init_before_training: bool = True,
) -> None:
    """Train Flow Matching model on collected PPO data.

    Args:
        data_path: Path to PPO data pickle file
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        validation_split: Fraction of data for validation
        max_samples: Maximum number of samples to use (None for all)
        episode_length: Length of each episode for grouping
        reward_percentile: Keep episodes with reward >= this percentile (0-1)
        min_episode_reward: If set, keep episodes with total reward >= this value
        hybrid_sampling: Enable hybrid sampling (mix high-quality + random coverage)
        high_quality_ratio: In hybrid mode, ratio of high-quality samples (0-1)
        high_quality_percentile: In hybrid mode, percentile for high-quality (0-1)
        output_dir: Directory to save models
        seed: Random seed
        hidden_size: Size of hidden layers (default: 64)
        num_layers: Number of hidden layers (default: 4)
    """
    config = TrainingConfig().to_dict() | EnvConfig().to_dict()
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # When metrics_file is provided by the pipeline, only the parent process
    # owns W&B and this process emits JSONL events. Standalone execution keeps
    # direct W&B logging for backward compatibility.
    wandb_run = None
    if config["wandb_enabled"] and metrics_file is None:
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "wandb is required when wandb_enabled=True."
            ) from error
        wandb_run = wandb.init(
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            name=wandb_run_name or f"decoder_fm_stage_{stage}",
            id=wandb_run_id,
            resume="allow" if wandb_run_id else None,
            group=config["wandb_group"] or wandb_run_id,
            tags=list(config["wandb_tags"]),
            mode=config["wandb_mode"],
            config={**config, "pipeline_run_id": wandb_run_id},
        )
        wandb_run.define_metric("pipeline/decoder_step")
        wandb_run.define_metric("decoder/*", step_metric="pipeline/decoder_step")

    # Load data
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    resume_checkpoint = None
    resume_checkpoint_path = None
    if not stage_init_before_training:
        checkpoint_from_data = data.get("fm_model")
        if stage > 0 and not checkpoint_from_data:
            raise ValueError(
                "stage_init_before_training=False requires collected data with an "
                f"'fm_model' checkpoint path for stage {stage}."
            )

        if checkpoint_from_data:
            candidate_path = Path(checkpoint_from_data).expanduser()
            if not candidate_path.is_file():
                raise FileNotFoundError(
                    f"Decoder checkpoint referenced by collected data was not found: "
                    f"{candidate_path}"
                )
            with candidate_path.open("rb") as f:
                candidate_checkpoint = pickle.load(f)
            if not isinstance(candidate_checkpoint, dict):
                raise ValueError(
                    f"Decoder checkpoint must contain a dictionary: {candidate_path}"
                )

            # In a pure-online run, stage 0 data is collected with the identity
            # decoder. That checkpoint is only a bootstrap policy, not a trained
            # state to resume, so initialize the trainable FM network normally.
            # Offline stage-0 checkpoints and every later-stage checkpoint are
            # still restored directly.
            should_resume_decoder = stage > 0 or not candidate_checkpoint.get(
                "is_identity", False
            )
            if should_resume_decoder:
                required_fields = {
                    "params", "obs_stats", "config", "obs_dim", "action_dim"
                }
                missing_fields = sorted(
                    required_fields.difference(candidate_checkpoint)
                )
                if missing_fields:
                    raise ValueError(
                        f"Decoder resume checkpoint {candidate_path} is missing "
                        f"required fields: {missing_fields}"
                    )
                resume_checkpoint_path = candidate_path
                resume_checkpoint = candidate_checkpoint
                print(f"Continuing decoder training from: {resume_checkpoint_path}")
            else:
                print(
                    "Initializing decoder normally for online stage 0; identity "
                    "checkpoint is used only for data collection."
                )

    states = data["states"]
    actions = data["actions"]

    # Episode-based filtering using rewards
    if "rewards" in data and (config['fm_reward_percentile'] > 0 or config['fm_min_episode_reward'] is not None or config['fm_hybrid_sampling']):
        rewards = data["rewards"]

        # Group data by episodes
        n_episodes = len(states) // config['episode_length']
        if len(states) % config['episode_length'] != 0:
            trim_to = n_episodes * config['episode_length']
            states = states[:trim_to]
            actions = actions[:trim_to]
            rewards = rewards[:trim_to]

        episodes = []
        for i in range(n_episodes):
            start = i * config['episode_length']
            end = start + config['episode_length']
            episode_total_reward = rewards[start:end].sum()
            episodes.append({
                'idx': i,
                'start': start,
                'end': end,
                'total_reward': float(episode_total_reward)
            })

        episode_rewards = np.array([ep['total_reward'] for ep in episodes])

        # Apply filtering or hybrid sampling
        if config['fm_hybrid_sampling']:
            hq_threshold = np.percentile(episode_rewards, [config['fm_high_quality_percentile']] * 100)
            high_quality_episodes = [ep for ep in episodes if ep['total_reward'] >= hq_threshold]

            n_high_quality = int(len(episodes) * config['fm_high_quality_ratio'])
            n_coverage = len(episodes) - n_high_quality

            if len(high_quality_episodes) >= n_high_quality:
                sampled_hq = np.random.choice(len(high_quality_episodes), n_high_quality, replace=False)
                selected_hq_episodes = [high_quality_episodes[i] for i in sampled_hq]
            else:
                sampled_hq = np.random.choice(len(high_quality_episodes), n_high_quality, replace=True)
                selected_hq_episodes = [high_quality_episodes[i] for i in sampled_hq]

            sampled_coverage = np.random.choice(len(episodes), n_coverage, replace=False)
            selected_coverage_episodes = [episodes[i] for i in sampled_coverage]

            keep_episodes = selected_hq_episodes + selected_coverage_episodes

        elif config['fm_min_episode_reward'] is not None:
            keep_episodes = [ep for ep in episodes if ep['total_reward'] >= config['fm_min_episode_reward']]
        else:
            threshold = np.percentile(episode_rewards, config['fm_reward_percentile'] * 100)
            keep_episodes = [ep for ep in episodes if ep['total_reward'] >= threshold]

        # Rebuild data from kept episodes
        keep_indices = []
        for ep in keep_episodes:
            keep_indices.extend(range(ep['start'], ep['end']))

        states = states[keep_indices]
        actions = actions[keep_indices]
        rewards = rewards[keep_indices]
    else:
        if "rewards" in data:
            rewards = data["rewards"]

    # Optionally subsample data for faster training
    if config['fm_max_samples'] is not None and len(states) > config['fm_max_samples']:
        sample_indices = np.random.choice(len(states), config['fm_max_samples'], replace=False)
        states = states[sample_indices]
        actions = actions[sample_indices]
        if "rewards" in data:
            rewards = rewards[sample_indices]

    # Split data
    n_samples = len(states)
    n_train = int(n_samples * (1 - config['fm_validation_split']))
    indices = np.random.permutation(n_samples)

    train_states = states[indices[:n_train]]
    train_actions = actions[indices[:n_train]]
    val_states = states[indices[n_train:]]
    val_actions = actions[indices[n_train:]]

    # Initialize FM model
    obs_dim = states.shape[1]
    action_dim = actions.shape[1]

    # Build hidden dims from parameters
    hidden_dims = tuple([config['fm_hidden_size']] * config['fm_num_layers'])

    if resume_checkpoint is None:
        decoder_config = DecoderFMConfig(
            flow_steps=10,
            timestep_embed_dim=8,  # FPO uses 8
            hidden_dims=hidden_dims,  # Configurable network size
            policy_output_scale=1.0,  # Changed to 1.0 for supervised learning
            learning_rate=config['fm_learning_rate'],
            batch_size=config['fm_batch_size'],
            num_epochs=config['fm_num_epochs'],
            n_samples_per_action=config['fm_n_samples_per_action'],  # FPO's actual default
            normalize_observations=True,
            sde_sigma=0.0,
            feather_std=0.0,
        )
    else:
        if int(resume_checkpoint["obs_dim"]) != obs_dim:
            raise ValueError(
                f"Resume checkpoint obs_dim={resume_checkpoint['obs_dim']} does not "
                f"match collected data obs_dim={obs_dim}."
            )
        if int(resume_checkpoint["action_dim"]) != action_dim:
            raise ValueError(
                f"Resume checkpoint action_dim={resume_checkpoint['action_dim']} does "
                f"not match collected data action_dim={action_dim}."
            )
        decoder_config = resume_checkpoint["config"]
        if not isinstance(decoder_config, DecoderFMConfig):
            raise TypeError(
                f"Resume checkpoint {resume_checkpoint_path} has an invalid "
                f"decoder config type: {type(decoder_config).__name__}."
            )

    prng = jax.random.PRNGKey(config['seed'])
    fm_state = DecoderFMState.init(prng, obs_dim, action_dim, decoder_config)

    # Online optimizers are independent from offline training. Offline
    # checkpoints restore only model parameters and observation statistics;
    # DecoderFMState.init provides fresh online optimizer/PRNG/step state.
    resume_is_offline = bool(
        resume_checkpoint is not None
        and (
            resume_checkpoint.get("is_frozen_offline", False)
            or resume_checkpoint.get("offline_checkpoint_type") is not None
            or str(resume_checkpoint.get("checkpoint_format", "")).startswith(
                "gorl_offline"
            )
        )
    )
    with jdc.copy_and_mutate(fm_state) as fm_state:
        if resume_checkpoint is not None:
            fm_state.params = resume_checkpoint["params"]
            fm_state.obs_stats = resume_checkpoint["obs_stats"]
            # Offline optimizer state is never inherited. A previous online
            # stage may retain its own optimizer when continuation is requested.
            if not resume_is_offline and "fm_opt_state" in resume_checkpoint:
                fm_state.opt_state = resume_checkpoint["fm_opt_state"]
            if not resume_is_offline and "fm_prng" in resume_checkpoint:
                fm_state.prng = resume_checkpoint["fm_prng"]
            if not resume_is_offline and "fm_steps" in resume_checkpoint:
                fm_state.steps = resume_checkpoint["fm_steps"]
        fm_state.obs_stats = fm_state.obs_stats.update(jnp.array(train_states))

    # Preserve the encoder train state in decoder checkpoints. The pipeline only
    # passes the latest decoder checkpoint into the next encoder stage, so these
    # fields allow stage_init_before_training=False to resume PPO or RLPD.
    carried_encoder_fields = {}
    encoder_checkpoint_path = data.get("encoder_checkpoint") or data.get(
        "ppo_z_checkpoint"
    )
    if encoder_checkpoint_path:
        encoder_checkpoint_path = Path(encoder_checkpoint_path).expanduser()
        if not encoder_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Encoder checkpoint referenced by collected data was not found: "
                f"{encoder_checkpoint_path}"
            )
        with encoder_checkpoint_path.open("rb") as f:
            encoder_checkpoint = pickle.load(f)
        encoder_algorithm = data.get("encoder_algorithm")
        if encoder_algorithm is None:
            encoder_algorithm = (
                "rlpd" if "rlpd_z_actor_params" in encoder_checkpoint else "ppo"
            )
        if encoder_algorithm == "rlpd":
            required_encoder_fields = {
                "rlpd_z_actor_params",
                "rlpd_z_critic_params",
                "rlpd_z_target_critic_params",
                "rlpd_z_log_temperature",
                "rlpd_z_obs_stats",
            }
            encoder_field_names = (
                "rlpd_z_actor_params",
                "rlpd_z_critic_params",
                "rlpd_z_target_critic_params",
                "rlpd_z_log_temperature",
                "rlpd_z_actor_opt_state",
                "rlpd_z_critic_opt_state",
                "rlpd_z_temperature_opt_state",
                "rlpd_z_obs_stats",
                "rlpd_z_prng",
                "rlpd_z_steps",
            )
        elif encoder_algorithm == "ppo":
            required_encoder_fields = {"ppo_z_params", "ppo_z_obs_stats"}
            encoder_field_names = (
                "ppo_z_params",
                "ppo_z_obs_stats",
                "ppo_z_anchor_policy",
                "ppo_z_anchor_obs_stats",
                "ppo_z_opt_state",
                "ppo_z_prng",
                "ppo_z_steps",
            )
        else:
            raise ValueError(
                f"Unsupported encoder_algorithm in collected data: {encoder_algorithm!r}"
            )
        missing_encoder_fields = sorted(
            required_encoder_fields.difference(encoder_checkpoint)
        )
        if missing_encoder_fields:
            raise ValueError(
                f"Encoder checkpoint {encoder_checkpoint_path} is missing required "
                f"fields: {missing_encoder_fields}"
            )
        for key in encoder_field_names:
            if key in encoder_checkpoint:
                carried_encoder_fields[key] = encoder_checkpoint[key]

    # Training loop
    n_batches = n_train // config['fm_batch_size']
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0
    patience = 20  # Early stopping patience

    for epoch in range(config['fm_num_epochs']):
        # Training
        epoch_losses = []
        epoch_metrics = []

        # Shuffle training data
        perm = np.random.permutation(n_train)

        for batch_idx in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{config['fm_num_epochs']}"):
            # Get batch
            start_idx = batch_idx * config['fm_batch_size']
            end_idx = start_idx + config['fm_batch_size']
            batch_indices = perm[start_idx:end_idx]

            batch_obs = jnp.array(train_states[batch_indices])
            batch_actions = jnp.array(train_actions[batch_indices])

            # Training step
            fm_state, metrics = fm_state.train_step(batch_obs, batch_actions)

            epoch_losses.append(float(metrics["loss"]))
            epoch_metrics.append(metrics)

        # Compute epoch statistics
        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)

        # Validation
        val_batch_losses = []
        n_val_batches = min(50, len(val_states) // config['fm_batch_size'])  # Increased validation coverage

        for i in range(n_val_batches):
            start_idx = i * config['fm_batch_size']
            end_idx = start_idx + config['fm_batch_size']
            batch_obs = jnp.array(val_states[start_idx:end_idx])
            batch_actions = jnp.array(val_actions[start_idx:end_idx])

            # Use CFM loss for validation (same as training)
            # Normalize observations
            if fm_state.config.normalize_observations:
                obs_norm = (batch_obs - fm_state.obs_stats.mean) / (fm_state.obs_stats.std + 1e-8)
            else:
                obs_norm = batch_obs

            # Sample noise and timesteps
            prng_val_eps, prng_val_t, prng = jax.random.split(fm_state.prng, 3)
            val_eps = jax.random.normal(
                prng_val_eps,
                (config['fm_batch_size'], fm_state.config.n_samples_per_action, action_dim)
            )
            val_t = jax.random.uniform(
                prng_val_t,
                (config['fm_batch_size'], fm_state.config.n_samples_per_action, 1)
            )

            # Compute CFM loss
            cfm_loss = fm_state.compute_cfm_loss(
                obs_norm,
                batch_actions,
                val_eps,
                val_t
            )
            val_loss = jnp.mean(cfm_loss)
            val_batch_losses.append(float(val_loss))

        val_loss = np.mean(val_batch_losses)
        val_losses.append(val_loss)

        # Save best model
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            # Save checkpoint
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = output_path / f"fm_model_best_{timestamp}.pkl"

            checkpoint = {
                "params": fm_state.params,  # Only save parameters
                "obs_stats": fm_state.obs_stats,
                "fm_opt_state": fm_state.opt_state,
                "fm_prng": fm_state.prng,
                "fm_steps": fm_state.steps,
                "config": decoder_config,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "resumed_from": (
                    str(resume_checkpoint_path)
                    if resume_checkpoint_path is not None
                    else None
                ),
                **carried_encoder_fields,
            }

            with open(checkpoint_file, "wb") as f:
                pickle.dump(checkpoint, f)

            patience_counter = 0
        else:
            patience_counter += 1

        epoch_log = {
                "pipeline/decoder_step": global_epoch_offset + epoch + 1,
                "pipeline/stage": stage,
                "decoder/epoch": epoch + 1,
                "decoder/train_loss": float(train_loss),
                "decoder/val_loss": float(val_loss),
                "decoder/best_val_loss": float(best_val_loss),
                "decoder/patience_counter": patience_counter,
                "decoder/improved": int(improved),
        }
        if wandb_run is not None:
            wandb_run.log(epoch_log)
        elif metrics_file is not None:
            append_metrics(metrics_file, epoch_log)

        if patience_counter >= patience:
            break

    # Save final model
    final_file = output_path / f"fm_model_final_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    final_checkpoint = {
        "params": fm_state.params,  # Only save parameters
        "obs_stats": fm_state.obs_stats,
        "fm_opt_state": fm_state.opt_state,
        "fm_prng": fm_state.prng,
        "fm_steps": fm_state.steps,
        "config": decoder_config,
        "epoch": config['fm_num_epochs'],
        "train_loss": train_losses[-1],
        "val_loss": val_losses[-1],
        "train_history": train_losses,
        "val_history": val_losses,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "resumed_from": (
            str(resume_checkpoint_path)
            if resume_checkpoint_path is not None
            else None
        ),
        **carried_encoder_fields,
    }

    with open(final_file, "wb") as f:
        pickle.dump(final_checkpoint, f)

    completed_epoch = len(train_losses)
    completion_log = {
            "pipeline/decoder_step": global_epoch_offset + completed_epoch,
            "pipeline/stage": stage,
            "decoder/completed": 1,
            "decoder/completed_epochs": completed_epoch,
            "decoder/final_train_loss": float(train_losses[-1]),
            "decoder/final_val_loss": float(val_losses[-1]),
            "decoder/final_best_val_loss": float(best_val_loss),
            "stage/completed": 1,
            f"stage_{stage}/decoder_completed": 1,
            f"stage_{stage}/completed": 1,
    }
    if wandb_run is not None:
        wandb_run.log(completion_log)
        wandb_run.finish()
    elif metrics_file is not None:
        append_metrics(metrics_file, completion_log)

    print(f"Decoder (FM) done: loss={best_val_loss:.4f}, output={checkpoint_file}")


if __name__ == "__main__":
    tyro.cli(train_fm)