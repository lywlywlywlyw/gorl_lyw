from dataclasses import asdict, dataclass


@dataclass
class MeanFlowConfig:
    """MP1 one-step MeanFlow decoder settings."""
    meanflow_learning_rate: float = 3e-4
    meanflow_batch_size: int = 8192
    meanflow_num_epochs: int = 2_000_000
    meanflow_checkpoint_interval: int = 10_000
    meanflow_max_samples: int | None = None
    meanflow_min_epochs: int = 20
    meanflow_patience: int = 20
    meanflow_validation_split: float = 0.05
    meanflow_min_delta: float = 1e-4
    meanflow_eval_batches: int = 32
    # Match the Flow Matching decoder's 8-d timestep embedding and 64-wide
    # four-layer MLP capacity.
    meanflow_timestep_embed_dim: int = 8
    meanflow_hidden_dim: int = 64
    meanflow_num_res_blocks: int = 4
    meanflow_mlp_expansion: int = 2
    meanflow_policy_output_scale: float = 1.0
    meanflow_normalize_observations: bool = True
    meanflow_flow_ratio: float = 0.5
    meanflow_guidance_scale: float = 2.0
    use_dispersive: bool = False
    meanflow_dispersive_loss_weight: float = 0.5
    def to_dict(self) -> dict:
        return asdict(self)
