from dataclasses import dataclass, asdict


@dataclass
class FlowMatchingConfig:
    fm_batch_size: int = 8192
    fm_num_epochs: int = 50
    fm_learning_rate: float = 0.0003
    fm_max_samples: int = 10000000
    fm_hidden_size: int = 64
    fm_num_layers: int = 4
    fm_eval_episodes: int = 20

    # FM data filtering/sampling
    fm_hybrid_sampling: bool = False  # Disabled: use full random sampling
    fm_high_quality_ratio: float = 0.8
    fm_high_quality_percentile: float = 0.5

    fm_flow_steps: int = 10
    fm_timestep_embed_dim: int = 8
    fm_hidden_dims: tuple = (64, 64, 64, 64)  # Standard architecture
    fm_policy_output_scale: float = 1.0
    fm_n_samples_per_action: int = 8
    fm_normalize_observations: bool = True
    fm_sde_sigma: float = 0.0
    fm_feather_std: float = 0.0

    fm_reward_percentile: float = 0.0  
    fm_min_episode_reward: float | None = None  
    fm_hybrid_sampling: bool = False  
    fm_validation_split: float = 0.1
    fm_n_samples_per_action: int = 8

    def to_dict(self):
        return asdict(self)