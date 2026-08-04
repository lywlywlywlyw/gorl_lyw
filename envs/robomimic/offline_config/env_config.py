from dataclasses import dataclass, asdict


@dataclass
class EnvConfig:
    env_name: str = "Lift"
    dataset_path: str = "/root/GoRL/datasets/robomimic/low_dim.hdf5"
    action_repeat: int = 1
    episode_length: int = 300#1000
    num_envs: int = 96#50#16
    eval_num_envs: int = 10#50#32,

    def to_dict(self):
        return asdict(self)