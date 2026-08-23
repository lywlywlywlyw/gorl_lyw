from dataclasses import asdict, dataclass


@dataclass
class IQLConfig:
    """IQL encoder parameters matching run_offline_fm_frozen_new.py."""

    encoder_iql_steps: int = 500_000
    batch_size: int = 256
    # Match online RLPD so the warm-start critic has the same return scale.
    discount: float = 0.99
    expectile: float = 0.8
    temperature: float = 0.1
    max_adv_weight: float = 100.0
    target_update_rate: float = 0.005
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-4
    value_learning_rate: float = 1e-4
    max_grad_norm: float = 0.5
    q_hidden_size: int = 256
    q_hidden_layers: int = 2
    log_interval: int = 1000
    comparison_samples: int = 4096
    checkpoint_interval: int = 100_000
    validation_interval: int = 1_000
    validation_batches: int = 32
    early_stopping_min_steps: int = 50_000
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-3

    def to_dict(self) -> dict:
        return asdict(self)
