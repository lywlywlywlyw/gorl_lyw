from dataclasses import asdict, dataclass


@dataclass
class EnvConfig:
    """D4RL task settings shared by the online entry point."""
    dataset_path: str = "/root/GoRL/datasets/d4rl/hopper-medium-expert-v2_processed.hdf5"
    action_repeat: int = 1
    episode_length: int = 1000
    num_envs: int = 48
    eval_num_envs: int = 48
    dense_reward: bool = False
    success_reward_bonus: float = 0.0
    terminate_on_success: bool = False

    def to_dict(self):
        return asdict(self)
