from dataclasses import dataclass, asdict
@dataclass
class PPOConfig:
    ppo_batch_size: int = 256#1000#1024
    ppo_num_minibatches: int = 3#2#32
    ppo_num_updates_per_batch: int = 4#10#16
    ppo_unroll_length: int = 16#30
    ppo_learning_rate: float = 1e-3
    ppo_entropy_cost: float = 0
    ppo_discounting: float = 0.995

    # Training
    ppo_num_timesteps: int = 196608#393216#49152#60_000_000
    ppo_num_evals: int = 4#9#10

    # Normalization & Reward
    ppo_normalize_observations: bool = True
    ppo_reward_scaling: float = 10.0


    ppo_clipping_epsilon: float | None = None
    ppo_apply_tanh_in_rollout: bool = True
    ppo_z_regularization: float = 0.0
    latent_reg_coeff: float = 0.01#0.01
    latent_reg_threshold: float = 0.0
    ppo_max_grad_norm: float = 0.5
    ppo_use_tanh_jacobian_for_z: bool = False

    # Early stopping parameters
    ppo_eval_frequency: int = 600000#1000000
    ppo_min_steps: int = 1200000#0
    ppo_improvement_threshold: float | None = 15.0
    ppo_improvement_ratio_threshold: float | None = None
    ppo_improvement_window: int = 5
    ppo_reward_drop_threshold: float | None = 100.0
    ppo_reward_drop_ratio: float | None = None
    ppo_early_stopping: bool = False
    ppo_gae_lambda: float = 0.95
    ppo_normalize_advantage: bool = True
    ppo_value_loss_coeff: float = 0.25
    
    def to_dict(self):
        return asdict(self)