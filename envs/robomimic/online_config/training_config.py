from dataclasses import dataclass, asdict
from envs.robomimic.online_config.decoder_configs.fm_config import FlowMatchingConfig
from envs.robomimic.online_config.decoder_configs.meanflow_config import MeanFlowConfig
from envs.robomimic.online_config.encoder_configs.ppo_config import PPOConfig
from envs.robomimic.online_config.encoder_configs.rlpd_config import RLPDConfig

from typing import Annotated
import tyro
@dataclass
class TrainingConfig:
    # training overview config
    num_stages: Annotated[int, tyro.conf.arg(help="Number of training stages")] = 48
    seed: int = 1
    data_collection_iterations: int = 2#20
    z_regularization: float | None = None
    max_grad_norm: float = 0.5

    # encoder training config
    encoder_num_timesteps: Annotated[int, tyro.conf.arg(help="Default encoder training timesteps per stage")] = 2400000#100000000
    encoder_timesteps_per_stage: Annotated[str | None, tyro.conf.arg(help="Comma-separated timesteps for each stage (e.g., '60000000,60000000,30000000,30000000')")] =  ",".join(["393216"] * 48)#",".join(["196608"] * 96)#",".join(["49152"] * 384)#"4800000,4800000,2400000,2400000"#"60000000,60000000,30000000,30000000"
    encoder_type: str = "rlpd"

    # decoder training config
    decoder_type: str = "meanflow"#"flow_matching"

    # Weights & Biases logging
    wandb_enabled: bool = True
    wandb_project: str = "GoRL-robomimic"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    wandb_group: str | None = None
    wandb_tags: tuple[str, ...] = ("gorl", "fm", "robomimic")
    wandb_video_interval_evals: int = 1
    wandb_video_fps: int = 20
    wandb_video_width: int = 256
    wandb_video_height: int = 256
    wandb_video_frame_skip: int = 2
    q_gap_rollouts_per_state: int = 5

    def to_dict(self):
        if self.encoder_type == "ppo":
            encoder_config = PPOConfig().to_dict()
        elif self.encoder_type == "rlpd":
            encoder_config = RLPDConfig().to_dict()

            
        if self.decoder_type == "flow_matching":
            decoder_config = FlowMatchingConfig().to_dict()
        elif self.decoder_type == "meanflow":
            decoder_config = MeanFlowConfig().to_dict()
        else:
            raise ValueError("decoder_type must be 'flow_matching' or 'meanflow'.")
        return asdict(self) | encoder_config | decoder_config