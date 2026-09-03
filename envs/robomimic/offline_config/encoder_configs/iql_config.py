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
    # Offline-only KL regularization toward the decoder's N(0, I) latent
    # prior. This is intentionally independent of rlpd_latent_kl_weight.
    encoder_iql_prior_kl_weight: float = 0.0
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
    # Start model selection only after the critic scale has had time to form.
    early_stopping_min_steps: int = 100_000
    # Allow a long plateau before stopping; 50 validations = 50k updates.
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 1e-4
    # Small policy-quality tie-breaker in the normalized validation score.
    early_stopping_actor_nll_weight: float = 0.01
    # Optional post-IQL bridge finetune for compatibility with online RLPD.
    # The bridge uses a frozen copy of the IQL actor as a score teacher while
    # the live actor and critics continue from the early-stopped checkpoint.
    encoder_alignment_steps: int = 10_000
    encoder_score_matching_weight: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)
