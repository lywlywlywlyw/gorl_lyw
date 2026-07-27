"""Frozen-decoder offline training compatible with the online GoRL FM stack.

The method is the same two-stage procedure as ``run_offline_fm_frozen.py``:

1. Train a conditional flow-matching decoder to convergence and freeze it.
2. Invert dataset actions through the frozen decoder and train the latent
   encoder with IQL advantage-weighted behavior cloning.

This file is deliberately self-contained with respect to the old offline
scripts.  It imports only the same production network/state implementations
used by ``scripts/run_gorl_fm.py`` and its components.

The final pickle has both:

* top-level ``params/obs_stats/config/obs_dim/action_dim`` fields accepted by
  ``scripts/components/train_encoder_ppo.py`` as an FM decoder checkpoint;
* ``ppo_z_params/ppo_z_obs_stats/config`` fields matching online encoder
  checkpoints and accepted by ``scripts/components/collect_data_fm.py``.

Example:
    python run_offline_fm_frozen_new.py \
      --env-name WalkerWalk --data-path /path/to/offline_buffer.pkl
"""

from __future__ import annotations

import datetime
import ast
import json
import pickle
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax_dataclasses as jdc
import numpy as np
import optax
import tyro
import gymnasium as gym
from jax import Array
from jax import numpy as jnp
from tqdm import trange

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from flow_policy import encoder_ppo, math_utils, networks
from flow_policy.decoder_fm import DecoderFMConfig, DecoderFMState


PyTree = Any


@dataclass
class FrozenOfflineConfig:
    """Configuration for offline frozen-FM training."""

    env_name: str | None = None
    data_path: str | None = None
    d4rl_dataset: str | None = "walker2d-medium-expert-v2"
    dataset_dir: str = "datasets/d4rl"
    output_dir: str = (
        "results/offline_fm_frozen_new_"
        + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    seed: int = 0
    max_samples: int | None = None

    # Legacy buffers containing only (s, a, r).
    infer_transitions: bool = False
    episode_length: int = 1000

    # Decoder: train once to convergence, then freeze permanently.
    decoder_learning_rate: float = 3e-4
    decoder_hidden_size: int = 64
    decoder_num_layers: int = 4
    decoder_batch_size: int = 8192
    decoder_max_epochs: int = 200
    decoder_min_epochs: int = 20
    decoder_patience: int = 20
    decoder_validation_fraction: float = 0.05
    decoder_min_delta: float = 1e-4
    decoder_eval_batches: int = 32
    flow_steps: int = 10
    latent_inverse_steps: int = 10
    n_fm_samples_per_action: int = 8

    # IQL encoder. The policy and value layouts are fixed by EncoderState.init.
    encoder_iql_steps: int = 500_000
    batch_size: int = 256
    discount: float = 0.99
    expectile: float = 0.8
    temperature: float = 0.1
    max_adv_weight: float = 100.0
    target_update_rate: float = 0.005
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    value_learning_rate: float = 3e-4
    q_hidden_size: int = 256
    q_hidden_layers: int = 2
    log_interval: int = 1000
    comparison_samples: int = 4096
    checkpoint_interval: int = 100_000

    # These become part of the exact online EncoderConfig saved in checkpoint.
    online_num_timesteps: int = 100_000_000
    online_clipping_epsilon: float = 0.15
    online_z_regularization: float = 0.0005
    online_max_grad_norm: float = 0.5
    online_use_tanh_jacobian_for_z: bool = False

    wandb_project: str = "offline-fm"
    wandb_entity: str | None = None
    wandb_group: str | None = "experiment-1"
    wandb_name: str | None = "frozen-seed-0_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_mode: str = "disabled"


@dataclass
class ReplayBuffer:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    masks: np.ndarray

    def __len__(self) -> int:
        return len(self.observations)


class WandbLogger:
    def __init__(self, config: FrozenOfflineConfig):
        self.run = None
        if config.wandb_mode == "disabled":
            return
        try:
            import wandb
        except ImportError as error:
            raise ImportError("W&B logging requires the wandb package.") from error
        self.run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            group=config.wandb_group,
            name=config.wandb_name,
            mode=config.wandb_mode,
            config={**asdict(config), "method": "frozen_decoder"},
            tags=["frozen_decoder", "online_compatible"],
        )

    def log(self, metrics: dict[str, float], step: int) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def first_present(data: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in data:
            return data[key]
    return None


def load_exorl_directory(path: Path) -> dict[str, np.ndarray]:
    episode_files = sorted(path.glob("episode_*.npz"))
    if not episode_files:
        candidates = sorted(path.glob("**/episode_*.npz"))
        parents = {candidate.parent for candidate in candidates}
        if len(parents) != 1:
            raise ValueError(
                f"{path} must resolve to exactly one ExORL buffer; "
                f"found {len(parents)}."
            )
        episode_files = candidates

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for episode_path in episode_files:
        with np.load(episode_path) as episode:
            required = {"observation", "action", "reward", "discount"}
            missing = required.difference(episode.files)
            if missing:
                raise KeyError(f"{episode_path} is missing {sorted(missing)}.")
            obs = np.asarray(episode["observation"], dtype=np.float32)
            action = np.asarray(episode["action"], dtype=np.float32)
            reward = np.asarray(episode["reward"], dtype=np.float32).reshape(-1)
            discount = np.asarray(episode["discount"], dtype=np.float32).reshape(-1)
            if not (len(obs) == len(action) == len(reward) == len(discount)):
                raise ValueError(f"Inconsistent lengths in {episode_path}.")
            if len(obs) < 2:
                continue
            observations.append(obs[:-1])
            actions.append(action[1:])
            rewards.append(reward[1:])
            next_observations.append(obs[1:])
            masks.append(discount[1:])
    if not observations:
        raise ValueError(f"No non-empty ExORL episodes found under {path}.")
    return {
        "observations": np.concatenate(observations),
        "actions": np.concatenate(actions),
        "rewards": np.concatenate(rewards),
        "next_observations": np.concatenate(next_observations),
        "masks": np.concatenate(masks),
    }


def load_hdf5(path: Path) -> dict[str, np.ndarray]:
    try:
        import h5py
    except ImportError as error:
        raise ImportError("Loading HDF5 datasets requires h5py.") from error

    data: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as file:
        def collect(name: str, item: Any) -> None:
            if isinstance(item, h5py.Dataset):
                data[name] = item[()]

        file.visititems(collect)

    required = {"observations", "actions", "rewards", "terminals"}
    if required.issubset(data) and "next_observations" not in data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        rewards = np.asarray(data["rewards"], dtype=np.float32).reshape(-1)
        terminals = np.asarray(data["terminals"], dtype=bool).reshape(-1)
        if not (
            len(observations) == len(actions) == len(rewards) == len(terminals)
        ):
            raise ValueError(f"Inconsistent D4RL array lengths in {path}.")
        if len(observations) < 2:
            raise ValueError(f"D4RL dataset {path} has fewer than two rows.")
        timeouts = data.get("timeouts")
        if timeouts is None:
            timeouts = np.zeros(len(observations), dtype=bool)
            episode_step = 0
            for index in range(len(observations)):
                final_timestep = episode_step == 999
                timeouts[index] = final_timestep
                if terminals[index] or final_timestep:
                    episode_step = 0
                else:
                    episode_step += 1
        else:
            timeouts = np.asarray(timeouts, dtype=bool).reshape(-1)
            if len(timeouts) != len(observations):
                raise ValueError(f"D4RL timeouts length does not match {path}.")
        keep = ~timeouts[:-1]
        return {
            "observations": observations[:-1][keep],
            "actions": actions[:-1][keep],
            "rewards": rewards[:-1][keep],
            "next_observations": observations[1:][keep],
            "terminals": terminals[:-1][keep].astype(np.float32),
        }
    return data


_D4RL_INFOS_URL = (
    "https://raw.githubusercontent.com/Farama-Foundation/D4RL/master/"
    "d4rl/infos.py"
)


def official_d4rl_urls() -> dict[str, str]:
    try:
        with urllib.request.urlopen(_D4RL_INFOS_URL, timeout=30) as response:
            source = response.read().decode("utf-8")
    except Exception as error:
        raise RuntimeError(
            "Could not fetch the official D4RL dataset catalog. Check network "
            f"access to {_D4RL_INFOS_URL}."
        ) from error
    tree = ast.parse(source, filename=_D4RL_INFOS_URL)
    urls: dict[str, str] | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "DATASET_URLS"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                raise RuntimeError("D4RL DATASET_URLS is not a dictionary.")
            urls = {str(key): str(url) for key, url in value.items()}
            break
    if urls is None:
        raise RuntimeError("Could not find DATASET_URLS in the D4RL catalog.")
    locomotion = ("halfcheetah", "hopper", "walker2d", "ant")
    qualities = (
        "random", "medium", "expert", "medium-replay", "full-replay",
        "medium-expert",
    )
    for environment in locomotion:
        for quality in qualities:
            quality_file = quality.replace("-", "_")
            for version in (1, 2):
                dataset_id = f"{environment}-{quality}-v{version}"
                filename = f"{environment}_{quality_file}-v{version}.hdf5"
                urls[dataset_id] = (
                    "http://rail.eecs.berkeley.edu/datasets/offline_rl/"
                    f"gym_mujoco_v{version}/{filename}"
                )
    return urls


def download_d4rl_dataset(dataset_id: str, dataset_dir: Path) -> Path:
    if not dataset_id or Path(dataset_id).name != dataset_id:
        raise ValueError("d4rl_dataset must be a D4RL task ID, not a path.")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    destination = dataset_dir / f"{dataset_id}.hdf5"
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Using cached D4RL dataset: {destination}")
        return destination
    urls = official_d4rl_urls()
    if dataset_id not in urls:
        raise ValueError(f"Unknown D4RL dataset {dataset_id!r}.")
    url = urls[dataset_id].replace("http://", "https://", 1)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {dataset_id} from {url}")

    def report(blocks: int, block_size: int, total_size: int) -> None:
        downloaded = blocks * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            print(
                f"\r  {min(downloaded, total_size) / 2**20:.1f}/"
                f"{total_size / 2**20:.1f} MiB ({percent:.1f}%)",
                end="",
                flush=True,
            )

    try:
        urllib.request.urlretrieve(url, temporary, reporthook=report)
        print()
        if temporary.stat().st_size == 0:
            raise RuntimeError("downloaded file is empty")
        temporary.replace(destination)
    except Exception as error:
        if temporary.exists():
            temporary.unlink()
        raise RuntimeError(
            f"Failed to download {dataset_id} from {url}."
        ) from error
    print(f"Saved D4RL dataset to {destination}")
    return destination


def resolve_data_path(config: FrozenOfflineConfig) -> Path:
    if config.data_path is not None and config.d4rl_dataset is not None:
        raise ValueError("Use either --data-path or --d4rl-dataset, not both.")
    if config.d4rl_dataset is not None:
        dataset_dir = Path(config.dataset_dir).expanduser()
        if not dataset_dir.is_absolute():
            dataset_dir = Path(__file__).resolve().parent / dataset_dir
        return download_d4rl_dataset(config.d4rl_dataset, dataset_dir)
    if config.data_path is None:
        raise ValueError("Provide --data-path or --d4rl-dataset.")
    return Path(config.data_path).expanduser()


def load_replay_buffer(config: FrozenOfflineConfig) -> ReplayBuffer:
    path = resolve_data_path(config)
    if path.is_dir():
        data = load_exorl_directory(path)
    elif path.suffix.lower() in {".h5", ".hdf5"}:
        data = load_hdf5(path)
    else:
        with open(path, "rb") as file:
            data = pickle.load(file)
    if not isinstance(data, dict):
        raise TypeError("Offline dataset must be a dictionary.")

    observations = first_present(data, "observations", "states", "obs")
    actions = first_present(data, "actions", "action")
    rewards = first_present(data, "rewards", "reward")
    next_observations = first_present(
        data, "next_observations", "next_states", "next_obs"
    )
    masks = first_present(data, "masks", "discounts")
    terminals = first_present(data, "terminals", "dones", "done")
    timeouts = first_present(data, "timeouts", "truncations", "truncated")
    if observations is None or actions is None or rewards is None:
        raise KeyError("Dataset needs observations/states, actions, and rewards.")

    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("Observations and actions must be rank-2 arrays.")

    if next_observations is None:
        if not config.infer_transitions:
            raise KeyError(
                "Dataset lacks next observations; enable --infer-transitions "
                "only for fixed-length, episode-ordered data."
            )
        usable = (len(observations) // config.episode_length) * config.episode_length
        if config.episode_length < 2 or usable == 0:
            raise ValueError("Cannot infer transitions with this episode length.")
        observations = observations[:usable]
        actions = actions[:usable]
        rewards = rewards[:usable]
        next_observations = np.roll(observations, -1, axis=0)
        inferred_terminal = np.zeros(usable, dtype=np.float32)
        inferred_terminal[config.episode_length - 1 :: config.episode_length] = 1
        terminal_rows = inferred_terminal.astype(bool)
        next_observations[terminal_rows] = observations[terminal_rows]
        terminals = inferred_terminal
    else:
        next_observations = np.asarray(next_observations, dtype=np.float32)

    size = len(observations)
    if not all(len(array) == size for array in (actions, rewards, next_observations)):
        raise ValueError("All transition arrays must have equal lengths.")
    if masks is not None:
        masks = np.asarray(masks, dtype=np.float32).reshape(-1)
    elif terminals is not None:
        terminal_array = np.asarray(terminals, dtype=np.float32).reshape(-1)
        if timeouts is not None:
            terminal_array *= 1.0 - np.asarray(
                timeouts, dtype=np.float32
            ).reshape(-1)
        masks = 1.0 - terminal_array
    else:
        masks = np.ones(size, dtype=np.float32)
    if len(masks) != size:
        raise ValueError("Masks/terminals must match transition count.")

    if config.max_samples is not None and size > config.max_samples:
        rng = np.random.default_rng(config.seed)
        indices = rng.choice(size, config.max_samples, replace=False)
        observations = observations[indices]
        actions = actions[indices]
        rewards = rewards[indices]
        next_observations = next_observations[indices]
        masks = masks[indices]
    return ReplayBuffer(
        observations, actions, rewards, next_observations, masks
    )


def validate_config(config: FrozenOfflineConfig) -> None:
    if config.decoder_max_epochs < 1:
        raise ValueError("decoder_max_epochs must be positive.")
    if not 1 <= config.decoder_min_epochs <= config.decoder_max_epochs:
        raise ValueError("decoder_min_epochs must be within decoder epochs.")
    if config.decoder_patience < 1:
        raise ValueError("decoder_patience must be positive.")
    if not 0.0 < config.decoder_validation_fraction < 1.0:
        raise ValueError("decoder_validation_fraction must be in (0, 1).")
    if min(
        config.batch_size,
        config.decoder_batch_size,
        config.encoder_iql_steps,
        config.latent_inverse_steps,
        config.comparison_samples,
        config.checkpoint_interval,
    ) < 1:
        raise ValueError("Batch sizes and step/sample counts must be positive.")
    if config.wandb_mode not in {"online", "offline", "disabled"}:
        raise ValueError("wandb_mode must be online, offline, or disabled.")


def d4rl_gymnasium_env_id(dataset_id: str) -> str:
    """Map a D4RL MuJoCo dataset ID to its Gymnasium environment."""
    environment = dataset_id.split("-", 1)[0].lower()
    mapping = {
        "walker2d": "Walker2d-v5",
        "halfcheetah": "HalfCheetah-v5",
        "hopper": "Hopper-v5",
        "ant": "Ant-v5",
    }
    if environment not in mapping:
        raise ValueError(
            f"D4RL dataset {dataset_id!r} has no Gymnasium environment mapping. "
            f"Supported D4RL MuJoCo tasks: {sorted(mapping)}."
        )
    return mapping[environment]


def make_dataset_environment(
    config: FrozenOfflineConfig,
) -> tuple[Any, str]:
    """Create the environment corresponding to the selected offline dataset."""
    if config.env_name is not None:
        environment_id = config.env_name
    elif config.d4rl_dataset is not None:
        environment_id = d4rl_gymnasium_env_id(config.d4rl_dataset)
    else:
        raise ValueError(
            "--env-name is required when loading a custom --data-path."
        )
    try:
        environment = gym.make(environment_id)
    except Exception as error:
        raise RuntimeError(
            f"Could not create Gymnasium environment {environment_id!r} for "
            "the offline dataset. Ensure Gymnasium MuJoCo dependencies are "
            "installed."
        ) from error
    return environment, environment_id


def make_encoder_config(
    config: FrozenOfflineConfig,
    action_dim: int,
    episode_length: int,
) -> encoder_ppo.EncoderConfig:
    """Build checkpoint metadata without depending on MuJoCo Playground."""
    return encoder_ppo.EncoderConfig(
        action_repeat=1,
        batch_size=config.batch_size,
        discounting=config.discount,
        entropy_cost=0.0,
        episode_length=episode_length,
        learning_rate=config.actor_learning_rate,
        normalize_observations=True,
        num_envs=1,
        num_evals=1,
        num_minibatches=1,
        num_timesteps=config.online_num_timesteps,
        num_updates_per_batch=1,
        reward_scaling=1.0,
        unroll_length=1,
        z_dim=action_dim,
        clipping_epsilon=config.online_clipping_epsilon,
        z_regularization=config.online_z_regularization,
        max_grad_norm=config.online_max_grad_norm,
        use_tanh_jacobian_for_z=config.online_use_tanh_jacobian_for_z,
    )


def split_indices(
    size: int, validation_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if size < 2:
        raise ValueError("At least two transitions are required.")
    permutation = rng.permutation(size)
    validation_size = min(size - 1, max(1, int(size * validation_fraction)))
    return permutation[validation_size:], permutation[:validation_size]


def inverse_fm_batch(
    decoder: DecoderFMState,
    observations: Array,
    actions: Array,
    num_steps: int,
) -> Array:
    obs_norm = (observations - decoder.obs_stats.mean) / (
        decoder.obs_stats.std + 1e-8
    )
    times = jnp.linspace(0.0, 1.0, num_steps + 1)

    def step(x_t: Array, pair: tuple[Array, Array]) -> tuple[Array, None]:
        current, following = pair
        t = jnp.full((*x_t.shape[:-1], 1), current)
        velocity = decoder.flow_forward(obs_norm, x_t, decoder.embed_timestep(t))
        return x_t + (following - current) * velocity, None

    latent, _ = jax.lax.scan(
        step, actions, (times[:-1], times[1:])
    )
    return latent


def forward_fm_batch(
    decoder: DecoderFMState, observations: Array, latents: Array
) -> Array:
    obs_norm = (observations - decoder.obs_stats.mean) / (
        decoder.obs_stats.std + 1e-8
    )
    schedule = decoder.get_schedule()

    def step(x_t: Array, pair: tuple[Array, Array]) -> tuple[Array, None]:
        current, following = pair
        t = jnp.full((*x_t.shape[:-1], 1), current)
        velocity = decoder.flow_forward(obs_norm, x_t, decoder.embed_timestep(t))
        return x_t + (following - current) * velocity, None

    actions, _ = jax.lax.scan(
        step, latents, (schedule.t_current, schedule.t_next)
    )
    return actions


def build_latent_targets(
    decoder: DecoderFMState,
    buffer: ReplayBuffer,
    batch_size: int,
    inverse_steps: int,
) -> np.ndarray:
    invert = jax.jit(
        lambda obs, act: inverse_fm_batch(decoder, obs, act, inverse_steps)
    )
    chunks = []
    for start in trange(
        0, len(buffer), batch_size, desc="Invert actions", leave=False
    ):
        end = min(start + batch_size, len(buffer))
        chunks.append(
            np.asarray(
                invert(
                    jnp.asarray(buffer.observations[start:end]),
                    jnp.asarray(buffer.actions[start:end]),
                )
            )
        )
    return np.concatenate(chunks)


def decoder_validation_loss(
    decoder: DecoderFMState,
    buffer: ReplayBuffer,
    indices: np.ndarray,
    max_batches: int,
    key: Array,
) -> tuple[float, Array]:
    losses = []
    for batch_number, start in enumerate(
        range(0, len(indices), decoder.config.batch_size)
    ):
        if batch_number >= max_batches:
            break
        batch = indices[start : start + decoder.config.batch_size]
        obs = jnp.asarray(buffer.observations[batch])
        actions = jnp.asarray(buffer.actions[batch])
        obs_norm = (obs - decoder.obs_stats.mean) / (
            decoder.obs_stats.std + 1e-8
        )
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


def decoder_metrics(
    decoder: DecoderFMState,
    buffer: ReplayBuffer,
    indices: np.ndarray,
    inverse_steps: int,
) -> tuple[dict[str, float], np.ndarray]:
    obs = jnp.asarray(buffer.observations[indices])
    actions = jnp.asarray(buffer.actions[indices])
    latents = jax.jit(
        lambda o, a: inverse_fm_batch(decoder, o, a, inverse_steps)
    )(obs, actions)
    reconstructed = jax.jit(
        lambda o, z: forward_fm_batch(decoder, o, z)
    )(obs, latents)
    latent_std = jnp.std(latents, axis=0)
    metrics = {
        "decoder/cycle_action_mse": float(
            jnp.mean(jnp.square(reconstructed - actions))
        ),
        "decoder/cycle_action_mae": float(
            jnp.mean(jnp.abs(reconstructed - actions))
        ),
        "latent/mean_abs": float(jnp.mean(jnp.abs(jnp.mean(latents, axis=0)))),
        "latent/std_mean": float(jnp.mean(latent_std)),
        "latent/std_error": float(jnp.mean(jnp.abs(latent_std - 1.0))),
        "latent/norm_mean": float(jnp.mean(jnp.linalg.norm(latents, axis=-1))),
        "latent/max_abs": float(jnp.max(jnp.abs(latents))),
    }
    return metrics, np.asarray(latents)


def expectile_loss(diff: Array, expectile: float) -> Array:
    weight = jnp.where(diff > 0, expectile, 1.0 - expectile)
    return weight * jnp.square(diff)


def polyak_update(params: PyTree, targets: PyTree, tau: float) -> PyTree:
    return jax.tree.map(
        lambda param, target: tau * param + (1.0 - tau) * target,
        params,
        targets,
    )


def make_iql_update(
    config: FrozenOfflineConfig,
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
    value_optimizer: optax.GradientTransformation,
):
    @jax.jit
    def update(
        actor_params,
        actor_opt_state,
        q1_params,
        q2_params,
        critic_opt_state,
        value_params,
        value_opt_state,
        target_q1_params,
        target_q2_params,
        obs,
        actions,
        rewards,
        next_obs,
        masks,
        latent_actions,
    ):
        target_q = jnp.minimum(
            networks.q_mlp_fwd(target_q1_params, obs, actions),
            networks.q_mlp_fwd(target_q2_params, obs, actions),
        )

        def value_loss_fn(params):
            value = networks.value_mlp_fwd(params, obs)
            loss = jnp.mean(expectile_loss(target_q - value, config.expectile))
            return loss, (value, target_q - value)

        (value_loss, (value, advantage)), value_grads = jax.value_and_grad(
            value_loss_fn, has_aux=True
        )(value_params)
        value_updates, value_opt_state = value_optimizer.update(
            value_grads, value_opt_state, value_params
        )
        value_params = optax.apply_updates(value_params, value_updates)

        new_value = networks.value_mlp_fwd(value_params, obs)
        actor_advantage = jax.lax.stop_gradient(target_q - new_value)
        advantage_weight = jnp.minimum(
            jnp.exp(actor_advantage * config.temperature),
            config.max_adv_weight,
        )

        def actor_loss_fn(params):
            distribution = networks.gaussian_policy_fwd(params, obs)
            log_prob = jnp.sum(distribution.log_prob(latent_actions), axis=-1)
            return -jnp.mean(advantage_weight * log_prob)

        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(actor_params)
        actor_updates, actor_opt_state = actor_optimizer.update(
            actor_grads, actor_opt_state, actor_params
        )
        actor_params = optax.apply_updates(actor_params, actor_updates)

        next_value = jax.lax.stop_gradient(
            networks.value_mlp_fwd(value_params, next_obs)
        )
        bellman_target = rewards + config.discount * masks * next_value

        def critic_loss_fn(params):
            q1, q2 = params
            q1_value = networks.q_mlp_fwd(q1, obs, actions)
            q2_value = networks.q_mlp_fwd(q2, obs, actions)
            loss = jnp.mean(
                jnp.square(q1_value - bellman_target)
                + jnp.square(q2_value - bellman_target)
            )
            return loss, (q1_value, q2_value)

        (critic_loss, (q1_value, q2_value)), critic_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )((q1_params, q2_params))
        critic_updates, critic_opt_state = critic_optimizer.update(
            critic_grads, critic_opt_state, (q1_params, q2_params)
        )
        q1_params, q2_params = optax.apply_updates(
            (q1_params, q2_params), critic_updates
        )
        target_q1_params = polyak_update(
            q1_params, target_q1_params, config.target_update_rate
        )
        target_q2_params = polyak_update(
            q2_params, target_q2_params, config.target_update_rate
        )
        return (
            actor_params,
            actor_opt_state,
            q1_params,
            q2_params,
            critic_opt_state,
            value_params,
            value_opt_state,
            target_q1_params,
            target_q2_params,
            {
                "value_loss": value_loss,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "value": jnp.mean(value),
                "q1": jnp.mean(q1_value),
                "q2": jnp.mean(q2_value),
                "advantage": jnp.mean(advantage),
                "adv_weight": jnp.mean(advantage_weight),
            },
        )

    return update


def policy_metrics(
    actor_params: PyTree,
    q1_params: PyTree,
    q2_params: PyTree,
    value_params: PyTree,
    decoder: DecoderFMState,
    buffer: ReplayBuffer,
    normalized_observations: np.ndarray,
    indices: np.ndarray,
    latent_targets: np.ndarray,
) -> dict[str, float]:
    obs_norm = jnp.asarray(normalized_observations[indices])
    obs_raw = jnp.asarray(buffer.observations[indices])
    data_actions = jnp.asarray(buffer.actions[indices])
    targets = jnp.asarray(latent_targets[indices])
    distribution = networks.gaussian_policy_fwd(actor_params, obs_norm)
    policy_z = distribution.loc
    policy_actions = jax.jit(
        lambda o, z: forward_fm_batch(decoder, o, z)
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
    return {
        "comparison/policy_q": float(jnp.mean(q_policy)),
        "comparison/data_q": float(jnp.mean(q_data)),
        "comparison/policy_q_minus_data_q": float(jnp.mean(q_policy - q_data)),
        "comparison/policy_q_minus_v": float(jnp.mean(q_policy - value)),
        "comparison/policy_action_data_mse": float(
            jnp.mean(jnp.square(policy_actions - data_actions))
        ),
        "encoder/latent_nll": float(
            -jnp.mean(jnp.sum(distribution.log_prob(targets), axis=-1))
        ),
        "encoder/mean_target_mse": float(
            jnp.mean(jnp.square(policy_z - targets))
        ),
        "encoder/scale_mean": float(jnp.mean(distribution.scale)),
    }


def append_metrics(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a") as file:
        file.write(json.dumps(record) + "\n")


def save_compatible_checkpoint(
    path: Path,
    config: FrozenOfflineConfig,
    online_config: encoder_ppo.EncoderConfig,
    decoder: DecoderFMState,
    encoder_params: encoder_ppo.ActorCriticParams,
    encoder_obs_stats: Any,
    q1_params: PyTree,
    q2_params: PyTree,
    value_params: PyTree,
) -> None:
    obs_dim = int(decoder.obs_stats.mean.shape[-1])
    action_dim = int(decoder.params[-1][0].shape[-1])
    checkpoint = {
        # Standalone FM schema loaded by train_encoder_ppo.py.
        "params": decoder.params,
        "obs_stats": decoder.obs_stats,
        "config": decoder.config,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "epoch": config.decoder_max_epochs,
        "is_frozen_offline": True,
        # Combined online encoder schema loaded by collect_data_fm.py.
        "ppo_z_params": encoder_params,
        "ppo_z_obs_stats": encoder_obs_stats,
        "env_name": config.env_name,
        "decoder_type": "fm",
        "z_dim": action_dim,
        "fm_params": decoder.params,
        "fm_obs_stats": decoder.obs_stats,
        "online_encoder_config": online_config,
        # Offline-only training state/metadata.
        "offline_config": asdict(config),
        "q1_params": q1_params,
        "q2_params": q2_params,
        "value_params": value_params,
    }
    # collect_data_fm.py interprets "config" as EncoderConfig, while
    # train_encoder_ppo.py interprets it as DecoderFMConfig. One file cannot
    # place both types under the same key. The pipeline consumes this final
    # checkpoint as a decoder, so "config" stays DecoderFMConfig. A separate
    # encoder checkpoint is emitted below for collect-data/resume use.
    with open(path, "wb") as file:
        pickle.dump(checkpoint, file)


def save_encoder_checkpoint(
    path: Path,
    config: FrozenOfflineConfig,
    online_config: encoder_ppo.EncoderConfig,
    decoder: DecoderFMState,
    encoder_params: encoder_ppo.ActorCriticParams,
    encoder_obs_stats: Any,
) -> None:
    checkpoint = {
        "ppo_z_params": encoder_params,
        "ppo_z_obs_stats": encoder_obs_stats,
        "config": online_config,
        "env_name": config.env_name,
        "decoder_type": "fm",
        "iteration": 0,
        "reward": float("-inf"),
        "z_dim": int(decoder.params[-1][0].shape[-1]),
        "fm_params": decoder.params,
        "fm_obs_stats": decoder.obs_stats,
    }
    with open(path, "wb") as file:
        pickle.dump(checkpoint, file)


def main(config: FrozenOfflineConfig) -> None:
    validate_config(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    with open(output_dir / "config.json", "w") as file:
        json.dump(
            {**asdict(config), "method": "frozen_decoder"},
            file,
            indent=2,
        )

    logger = WandbLogger(config)
    rng = np.random.default_rng(config.seed)
    try:
        buffer = load_replay_buffer(config)
        env, environment_id = make_dataset_environment(config)
        config.env_name = environment_id
        obs_dim = buffer.observations.shape[-1]
        action_dim = buffer.actions.shape[-1]
        if env.observation_space.shape != (obs_dim,):
            raise ValueError(
                f"Dataset observation dim {obs_dim} does not match "
                f"{environment_id} observation space "
                f"{env.observation_space.shape}."
            )
        if env.action_space.shape != (action_dim,):
            raise ValueError(
                f"Dataset action dim {action_dim} does not match "
                f"{environment_id} action space {env.action_space.shape}."
            )
        episode_length = int(
            env.spec.max_episode_steps
            if env.spec is not None and env.spec.max_episode_steps is not None
            else config.episode_length
        )
        online_config = make_encoder_config(
            config, action_dim, episode_length
        )
        with open(output_dir / "config.json", "w") as file:
            json.dump(
                {
                    **asdict(config),
                    "method": "frozen_decoder",
                    "resolved_environment": environment_id,
                },
                file,
                indent=2,
            )

        key = jax.random.key(config.seed)
        key, decoder_key, actor_key, value_key, q1_key, q2_key = (
            jax.random.split(key, 6)
        )
        decoder_config = DecoderFMConfig(
            flow_steps=config.flow_steps,
            timestep_embed_dim=8,
            hidden_dims=(config.decoder_hidden_size,)
            * config.decoder_num_layers,
            policy_output_scale=1.0,
            learning_rate=config.decoder_learning_rate,
            batch_size=config.decoder_batch_size,
            num_epochs=config.decoder_max_epochs,
            n_samples_per_action=config.n_fm_samples_per_action,
            normalize_observations=True,
            sde_sigma=0.0,
            feather_std=0.0,
        )
        decoder = DecoderFMState.init(
            decoder_key, obs_dim, action_dim, decoder_config
        )
        with jdc.copy_and_mutate(decoder) as decoder:
            decoder.obs_stats = decoder.obs_stats.update(
                jnp.asarray(buffer.observations)
            )

        # Keep the same policy/value layer layouts used by EncoderState.init,
        # but size their inputs from the matching D4RL/Gym environment.
        actor_params = networks.mlp_init(
            actor_key, (obs_dim, 32, 32, 32, 32, action_dim * 2)
        )
        value_params = networks.mlp_init(
            value_key, (obs_dim, 256, 256, 256, 256, 256, 1)
        )
        encoder_obs_stats = math_utils.RunningStats.init((obs_dim,)).update(
            jnp.asarray(buffer.observations)
        )
        obs_mean = np.asarray(encoder_obs_stats.mean)
        obs_std = np.asarray(encoder_obs_stats.std)
        normalized_observations = (
            buffer.observations - obs_mean
        ) / (obs_std + 1e-8)
        normalized_next_observations = (
            buffer.next_observations - obs_mean
        ) / (obs_std + 1e-8)

        q_dims = (
            obs_dim + action_dim,
            *((config.q_hidden_size,) * config.q_hidden_layers),
            1,
        )
        q1_params = networks.mlp_init(q1_key, q_dims)
        q2_params = networks.mlp_init(q2_key, q_dims)
        train_indices, validation_indices = split_indices(
            len(buffer), config.decoder_validation_fraction, rng
        )
        comparison_indices = rng.choice(
            validation_indices,
            min(config.comparison_samples, len(validation_indices)),
            replace=False,
        )
        print(
            f"Loaded {len(buffer):,} transitions for {environment_id}. "
            "Training decoder once, then freezing it."
        )

        best_params = jax.tree.map(jnp.copy, decoder.params)
        best_validation = float("inf")
        stale_epochs = 0
        global_step = 0
        previous_latents: np.ndarray | None = None
        for epoch in trange(config.decoder_max_epochs, desc="Decoder epochs"):
            losses = []
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
                decoder,
                buffer,
                validation_indices,
                config.decoder_eval_batches,
                key,
            )
            comparison, current_latents = decoder_metrics(
                decoder,
                buffer,
                comparison_indices,
                config.latent_inverse_steps,
            )
            comparison["latent/drift_mse"] = (
                0.0
                if previous_latents is None
                else float(np.mean(np.square(current_latents - previous_latents)))
            )
            previous_latents = current_latents
            record = {
                "method": "frozen_decoder",
                "phase": "decoder",
                "global_step": global_step,
                "decoder/epoch": epoch + 1,
                "decoder/train_cfm_loss": float(np.mean(losses)),
                "decoder/validation_cfm_loss": validation_loss,
                **comparison,
            }
            append_metrics(metrics_path, record)
            logger.log(
                {k: v for k, v in record.items() if isinstance(v, (int, float))},
                global_step,
            )
            if validation_loss < best_validation - config.decoder_min_delta:
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
        latent_targets = build_latent_targets(
            decoder,
            buffer,
            config.decoder_batch_size,
            config.latent_inverse_steps,
        )
        frozen_metrics, _ = decoder_metrics(
            decoder,
            buffer,
            comparison_indices,
            config.latent_inverse_steps,
        )
        frozen_record = {
            "method": "frozen_decoder",
            "phase": "decoder_frozen",
            "global_step": global_step,
            "decoder/best_validation_cfm_loss": best_validation,
            **frozen_metrics,
        }
        append_metrics(metrics_path, frozen_record)
        logger.log(
            {
                k: v
                for k, v in frozen_record.items()
                if isinstance(v, (int, float))
            },
            global_step,
        )

        actor_optimizer = optax.adam(config.actor_learning_rate)
        critic_optimizer = optax.adam(config.critic_learning_rate)
        value_optimizer = optax.adam(config.value_learning_rate)
        actor_opt_state = actor_optimizer.init(actor_params)
        critic_opt_state = critic_optimizer.init((q1_params, q2_params))
        value_opt_state = value_optimizer.init(value_params)
        target_q1_params = jax.tree.map(jnp.copy, q1_params)
        target_q2_params = jax.tree.map(jnp.copy, q2_params)
        iql_update = make_iql_update(
            config, actor_optimizer, critic_optimizer, value_optimizer
        )
        accumulators: dict[str, list[float]] = {}
        policy_indices = rng.choice(
            len(buffer),
            min(config.comparison_samples, len(buffer)),
            replace=False,
        )
        for step in trange(config.encoder_iql_steps, desc="IQL"):
            indices = rng.integers(0, len(buffer), size=config.batch_size)
            (
                actor_params,
                actor_opt_state,
                q1_params,
                q2_params,
                critic_opt_state,
                value_params,
                value_opt_state,
                target_q1_params,
                target_q2_params,
                metrics,
            ) = iql_update(
                actor_params,
                actor_opt_state,
                q1_params,
                q2_params,
                critic_opt_state,
                value_params,
                value_opt_state,
                target_q1_params,
                target_q2_params,
                jnp.asarray(normalized_observations[indices]),
                jnp.asarray(buffer.actions[indices]),
                jnp.asarray(buffer.rewards[indices]),
                jnp.asarray(normalized_next_observations[indices]),
                jnp.asarray(buffer.masks[indices]),
                jnp.asarray(latent_targets[indices]),
            )
            for name, value in metrics.items():
                accumulators.setdefault(name, []).append(float(value))
            if (
                (step + 1) % config.log_interval == 0
                or step + 1 == config.encoder_iql_steps
            ):
                record = {
                    "method": "frozen_decoder",
                    "phase": "encoder",
                    "global_step": global_step + step + 1,
                    "encoder/iql_step": step + 1,
                    **{
                        f"iql/{name}": float(np.mean(values))
                        for name, values in accumulators.items()
                    },
                    **policy_metrics(
                        actor_params,
                        q1_params,
                        q2_params,
                        value_params,
                        decoder,
                        buffer,
                        normalized_observations,
                        policy_indices,
                        latent_targets,
                    ),
                }
                append_metrics(metrics_path, record)
                logger.log(
                    {
                        k: v
                        for k, v in record.items()
                        if isinstance(v, (int, float))
                    },
                    global_step + step + 1,
                )
                print(f"  step {step + 1}: {record}")
                accumulators.clear()

            completed_iql_steps = step + 1
            if completed_iql_steps % config.checkpoint_interval == 0:
                periodic_encoder_params = encoder_ppo.ActorCriticParams(
                    policy=actor_params, value=value_params
                )
                periodic_decoder_path = (
                    output_dir
                    / f"checkpoint_step_{completed_iql_steps:09d}.pkl"
                )
                periodic_encoder_path = (
                    output_dir
                    / f"encoder_checkpoint_step_{completed_iql_steps:09d}.pkl"
                )
                save_compatible_checkpoint(
                    periodic_decoder_path,
                    config,
                    online_config,
                    decoder,
                    periodic_encoder_params,
                    encoder_obs_stats,
                    q1_params,
                    q2_params,
                    value_params,
                )
                save_encoder_checkpoint(
                    periodic_encoder_path,
                    config,
                    online_config,
                    decoder,
                    periodic_encoder_params,
                    encoder_obs_stats,
                )
                print(
                    "Saved periodic checkpoints at IQL step "
                    f"{completed_iql_steps}: {periodic_decoder_path}, "
                    f"{periodic_encoder_path}"
                )

        trained_encoder_params = encoder_ppo.ActorCriticParams(
            policy=actor_params, value=value_params
        )
        decoder_path = output_dir / "checkpoint_final.pkl"
        encoder_path = output_dir / "encoder_checkpoint_final.pkl"
        save_compatible_checkpoint(
            decoder_path,
            config,
            online_config,
            decoder,
            trained_encoder_params,
            encoder_obs_stats,
            q1_params,
            q2_params,
            value_params,
        )
        save_encoder_checkpoint(
            encoder_path,
            config,
            online_config,
            decoder,
            trained_encoder_params,
            encoder_obs_stats,
        )
        print(f"Saved online-compatible decoder checkpoint: {decoder_path}")
        print(f"Saved online-compatible encoder checkpoint: {encoder_path}")
    finally:
        logger.finish()


if __name__ == "__main__":
    main(tyro.cli(FrozenOfflineConfig))
