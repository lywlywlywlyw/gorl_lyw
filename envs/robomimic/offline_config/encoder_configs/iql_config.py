from dataclasses import asdict, dataclass


@dataclass
class IQLConfig:
    """IQL encoder parameters matching run_offline_fm_frozen_new.py."""

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

    def to_dict(self) -> dict:
        return asdict(self)
