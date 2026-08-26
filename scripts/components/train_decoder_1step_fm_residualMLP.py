"""Train one-step MeanFlow model to learn PPO action distribution."""

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
from flow_policy.decoder_1step_fm_residualMLP import (
    Decoder1StepFMConfig,
    Decoder1StepFMState,
)


def train_fm(
    data_path: str = "data/ppo_training_data_WalkerWalk_20250928_212057.pkl",
    num_epochs: int = 80,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    validation_split: float = 0.1,
    max_samples: int | None = 10000000,
    episode_length: int = 1000,
    reward_percentile: float = 0.0,
    min_episode_reward: float | None = None,
    hybrid_sampling: bool = False,
    high_quality_ratio: float = 0.8,
    high_quality_percentile: float = 0.5,
    output_dir: str = "fm_models",
    seed: int = 42,
    timestep_embed_dim: int = 128,
    hidden_dim: int = 512,
    num_res_blocks: int = 4,
    mlp_expansion: int = 2,
) -> None:
    """Train one-step MeanFlow decoder on collected PPO data."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    states = data["states"]
    actions = data["actions"]

    if "rewards" in data and (reward_percentile > 0 or min_episode_reward is not None or hybrid_sampling):
        rewards = data["rewards"]

        n_episodes = len(states) // episode_length
        if len(states) % episode_length != 0:
            trim_to = n_episodes * episode_length
            states = states[:trim_to]
            actions = actions[:trim_to]
            rewards = rewards[:trim_to]

        episodes = []
        for i in range(n_episodes):
            start = i * episode_length
            end = start + episode_length
            episode_total_reward = rewards[start:end].sum()
            episodes.append({
                "idx": i,
                "start": start,
                "end": end,
                "total_reward": float(episode_total_reward),
            })

        episode_rewards = np.array([ep["total_reward"] for ep in episodes])

        if hybrid_sampling:
            hq_threshold = np.percentile(episode_rewards, high_quality_percentile * 100)
            high_quality_episodes = [
                ep for ep in episodes if ep["total_reward"] >= hq_threshold
            ]

            n_high_quality = int(len(episodes) * high_quality_ratio)
            n_coverage = len(episodes) - n_high_quality

            if len(high_quality_episodes) >= n_high_quality:
                sampled_hq = rng.choice(
                    len(high_quality_episodes),
                    n_high_quality,
                    replace=False,
                )
            else:
                sampled_hq = rng.choice(
                    len(high_quality_episodes),
                    n_high_quality,
                    replace=True,
                )
            selected_hq_episodes = [high_quality_episodes[i] for i in sampled_hq]

            sampled_coverage = rng.choice(len(episodes), n_coverage, replace=False)
            selected_coverage_episodes = [episodes[i] for i in sampled_coverage]

            keep_episodes = selected_hq_episodes + selected_coverage_episodes
        elif min_episode_reward is not None:
            keep_episodes = [
                ep for ep in episodes if ep["total_reward"] >= min_episode_reward
            ]
        else:
            threshold = np.percentile(episode_rewards, reward_percentile * 100)
            keep_episodes = [
                ep for ep in episodes if ep["total_reward"] >= threshold
            ]

        keep_indices = []
        for ep in keep_episodes:
            keep_indices.extend(range(ep["start"], ep["end"]))

        states = states[keep_indices]
        actions = actions[keep_indices]
    elif "rewards" in data:
        rewards = data["rewards"]

    if max_samples is not None and len(states) > max_samples:
        sample_indices = rng.choice(len(states), max_samples, replace=False)
        states = states[sample_indices]
        actions = actions[sample_indices]
        if "rewards" in data:
            rewards = rewards[sample_indices]
            del rewards

    n_samples = len(states)
    n_train = int(n_samples * (1 - validation_split))
    indices = rng.permutation(n_samples)

    train_states = states[indices[:n_train]]
    train_actions = actions[indices[:n_train]]
    val_states = states[indices[n_train:]]
    val_actions = actions[indices[n_train:]]

    obs_dim = states.shape[1]
    action_dim = actions.shape[1]
    config = Decoder1StepFMConfig(
        flow_steps=1,
        timestep_embed_dim=timestep_embed_dim,
        hidden_dim=hidden_dim,
        num_res_blocks=num_res_blocks,
        mlp_expansion=mlp_expansion,
        condition_type="film",
        policy_output_scale=1.0,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        n_samples_per_action=1,
        normalize_observations=True,
        feather_std=0.0,
    )

    prng = jax.random.PRNGKey(seed)
    fm_state = Decoder1StepFMState.init(prng, obs_dim, action_dim, config)

    with jdc.copy_and_mutate(fm_state) as fm_state:
        fm_state.obs_stats = fm_state.obs_stats.update(jnp.array(train_states))

    n_batches = n_train // batch_size
    best_val_loss = float("inf")
    checkpoint_file = None
    train_losses = []
    val_losses = []
    patience_counter = 0
    patience = 20

    for epoch in range(num_epochs):
        epoch_losses = []
        perm = rng.permutation(n_train)

        for batch_idx in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{num_epochs}"):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_indices = perm[start_idx:end_idx]

            batch_obs = jnp.array(train_states[batch_indices])
            batch_actions = jnp.array(train_actions[batch_indices])

            fm_state, metrics = fm_state.train_step(batch_obs, batch_actions)
            epoch_losses.append(float(metrics["loss"]))

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        train_losses.append(train_loss)

        val_batch_losses = []
        n_val_batches = min(50, len(val_states) // batch_size)

        for i in range(n_val_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_obs = jnp.array(val_states[start_idx:end_idx])
            batch_actions = jnp.array(val_actions[start_idx:end_idx])
            obs_norm = fm_state._normalize_obs(batch_obs)

            prng_val_eps, prng_val_tr, prng = jax.random.split(prng, 3)
            val_eps = jax.random.normal(prng_val_eps, batch_actions.shape)
            val_t, val_r = fm_state.sample_t_r(prng_val_tr, batch_size)

            total_loss, _, _ = fm_state.compute_meanflow_loss(
                obs_norm,
                batch_actions,
                val_eps,
                val_t,
                val_r,
            )
            val_batch_losses.append(float(total_loss))

        val_loss = float(np.mean(val_batch_losses)) if val_batch_losses else train_loss
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = output_path / f"fm_1step_model_best_{timestamp}.pkl"

            checkpoint = {
                "params": fm_state.params,
                "obs_stats": fm_state.obs_stats,
                "config": config,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "is_1step_fm": True,
            }

            with open(checkpoint_file, "wb") as f:
                pickle.dump(checkpoint, f)

            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    final_file = output_path / f"fm_1step_model_final_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    final_checkpoint = {
        "params": fm_state.params,
        "obs_stats": fm_state.obs_stats,
        "config": config,
        "epoch": num_epochs,
        "train_loss": train_losses[-1],
        "val_loss": val_losses[-1],
        "train_history": train_losses,
        "val_history": val_losses,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "is_1step_fm": True,
    }

    with open(final_file, "wb") as f:
        pickle.dump(final_checkpoint, f)

    output_file = checkpoint_file if checkpoint_file is not None else final_file
    print(f"Decoder (1-step FM) done: loss={best_val_loss:.4f}, output={output_file}")


if __name__ == "__main__":
    tyro.cli(train_fm)
