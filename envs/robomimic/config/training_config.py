from dataclasses import dataclass, asdict
from envs.robomimic.config.decoder_configs.fm_config import FlowMatchingConfig
from envs.robomimic.config.encoder_configs.ppo_config import PPOConfig
from typing import Annotated
import tyro
@dataclass
class TrainingConfig:
    # training overview config
    num_stages: Annotated[int, tyro.conf.arg(help="Number of training stages")] = 4
    seed: int = 1
    data_collection_iterations: int = 20
    z_regularization: float | None = None
    max_grad_norm: float = 0.5,

    # encoder training config
    encoder_num_timesteps: Annotated[int, tyro.conf.arg(help="Default encoder training timesteps per stage")] = 2400000#100000000
    encoder_timesteps_per_stage: Annotated[str | None, tyro.conf.arg(help="Comma-separated timesteps for each stage (e.g., '60000000,60000000,30000000,30000000')")] =  "4800000,4800000,2400000,2400000"#"60000000,60000000,30000000,30000000"
    encoder_type = "ppo"

    # decoder training config
    decoder_type = "fm"

    def to_dict(self):
        if self.encoder_type == "ppo":
            encoder_config = PPOConfig().to_dict()
            
            
        if self.decoder_type == "fm":
            decoder_config = FlowMatchingConfig().to_dict()
        return asdict(self) | encoder_config | decoder_config