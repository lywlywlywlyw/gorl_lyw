from dataclasses import asdict, dataclass


@dataclass
class RLPDConfig:
    """RLPD encoder hyperparameters for latent-space online training."""

    rlpd_batch_size: int = 1024
    rlpd_replay_buffer_capacity: int = 200_000
    rlpd_learning_starts: int = 8192#24_576
    rlpd_updates_per_env_step: int = 1
    rlpd_policy_update_period: int = 4
    rlpd_actor_learning_rate: float = 1e-4
    rlpd_critic_learning_rate: float = 3e-4
    rlpd_temperature_learning_rate: float = 1e-4
    rlpd_learn_temperature: bool = False
    rlpd_discounting: float = 0.99
    rlpd_target_update_rate: float = 0.005
    rlpd_initial_temperature: float = 0.02
    rlpd_target_entropy: float | None = None
    rlpd_backup_entropy: bool = False
    rlpd_critic_ensemble_size: int = 2
    rlpd_critic_subsample_size: int = None
    rlpd_hidden_size: int = 256
    rlpd_hidden_layers: int = 2
    rlpd_reward_scaling: float = 1.0
    rlpd_reward_bias: float = 0.0
    rlpd_max_grad_norm: float = 10.0
    rlpd_latent_kl_weight: float = 1.0
    rlpd_normalize_observations: bool = True
    rlpd_apply_tanh_in_rollout: bool = False
    rlpd_rollout_steps_per_iteration: int = 8
    rlpd_num_evals: int = 10
    rlpd_offline_ratio: float = 0.5
    rlpd_offline_latent_dataset_path: str | None = None
    rlpd_log_interval: int = 10
    rlpd_checkpoint_interval: int = 100

    def to_dict(self) -> dict:
        return asdict(self)
