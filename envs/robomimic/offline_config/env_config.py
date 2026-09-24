from dataclasses import dataclass, asdict


@dataclass
class EnvConfig:
    env_name: str = "Lift"
    dataset_path: str = "/root/GoRL/datasets/robomimic/lift_mg_low_dim_dense_done_processed_v141_nocut.hdf5"
    action_repeat: int = 1
    episode_length: int = 150#1000
    num_envs: int = 96#50#16
    eval_num_envs: int = 10
    dense_reward: bool = True

    def to_dict(self):
        return asdict(self)
