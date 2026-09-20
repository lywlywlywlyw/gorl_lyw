import datetime
from dataclasses import asdict, dataclass

from envs.robomimic.offline_config.decoder_configs.fm_config import (
    FlowMatchingConfig,
)
from envs.robomimic.offline_config.decoder_configs.meanflow_config import MeanFlowConfig
from envs.robomimic.offline_config.encoder_configs.iql_config import IQLConfig


@dataclass
class TrainingConfig:
    """Top-level frozen-FM offline training configuration."""

    seed: int = 0
    environment: str = "robomimic"
    dataset_path: str | None = None
    output_dir: str = (
        "results/offline_fm_frozen_robomimic_"
        + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    decoder_type: str = "meanflow"#"flow_matching"
    encoder_type: str = "iql"
    # Optional offline checkpoint produced by run_offline_fm_frozen_robomimic.py.
    # Decoder-only checkpoints resume decoder training; combined checkpoints
    # resume directly from the encoder stage.
    checkpoint_path: str | None = None

    # Parameters saved in the online-compatible EncoderConfig. These mirror
    # FrozenOfflineConfig in run_offline_fm_frozen_new.py.
    online_num_timesteps: int = 100_000_000
    online_clipping_epsilon: float = 0.15
    online_z_regularization: float = 0.0005
    online_max_grad_norm: float = 0.5
    online_use_tanh_jacobian_for_z: bool = False

    # Weights & Biases logging
    wandb_enabled: bool = True
    wandb_project: str = "offline-fm"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_name: str | None = (
        "frozen-seed-0_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    wandb_mode: str = "online"

    def to_dict(self) -> dict:
        return (
            asdict(self)
            | IQLConfig().to_dict()
            | FlowMatchingConfig().to_dict()
            | MeanFlowConfig().to_dict()
        )