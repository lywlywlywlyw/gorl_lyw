from dataclasses import dataclass, asdict


@dataclass
class EnvConfig:
    env_name: str = "Can"
    dataset_path: str = "/root/GoRL/datasets/robomimic/mg_can_low_dim_dense_done_processed_success_v141_nocut.hdf5"
    action_repeat: int = 1
    episode_length: int = 150#1000
    num_envs: int = 48#50#16
    eval_num_envs: int = 48
    dense_reward: bool = True
    success_reward_bonus: float = 0.0
    terminate_on_success: bool = False
    # Whether to keep the decoder fixed during online training.
    freeze_decoder: bool = False

    def to_dict(self):
        return asdict(self)
