from dataclasses import dataclass, asdict


@dataclass
class FlowMatchingConfig:
    """Frozen decoder parameters matching run_offline_fm_frozen_new.py."""

    fm_learning_rate: float = 3e-4
    fm_hidden_size: int = 64
    fm_num_layers: int = 4
    fm_batch_size: int = 8192
    fm_num_epochs: int = 200
    fm_max_samples: int | None = None
    fm_min_epochs: int = 20
    fm_patience: int = 20
    fm_validation_split: float = 0.05
    fm_min_delta: float = 1e-4
    fm_eval_batches: int = 32
    fm_flow_steps: int = 10
    fm_latent_inverse_steps: int = 10
    fm_timestep_embed_dim: int = 8
    fm_policy_output_scale: float = 1.0
    fm_n_samples_per_action: int = 8
    fm_normalize_observations: bool = True
    fm_sde_sigma: float = 0.0
    fm_feather_std: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)