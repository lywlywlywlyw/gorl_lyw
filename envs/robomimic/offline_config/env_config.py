from dataclasses import dataclass, asdict


@dataclass
class EnvConfig:
    env_name: str = "Lift"
    dataset_path: str = "/root/GoRL/datasets/robomimic/mg_lift_low_dim_dense_done_processed.hdf5"
    action_repeat: int = 1
    episode_length: int = 150#1000
    num_envs: int = 96#50#16
    eval_num_envs: int = 48
    dense_reward: bool = True

    def to_dict(self):
        return asdict(self)
