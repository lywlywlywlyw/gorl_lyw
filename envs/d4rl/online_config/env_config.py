from dataclasses import asdict, dataclass


@dataclass
class EnvConfig:
    """Minari dataset and Gymnasium v5 task settings."""
    dataset_path: str = "mujoco/halfcheetah/medium-v0"
    action_repeat: int = 1
    episode_length: int = 1000
    num_envs: int = 48
    eval_num_envs: int = 48
    dense_reward: bool = False
    success_reward_bonus: float = 0.0
    terminate_on_success: bool = False

    def to_dict(self):
        return asdict(self)
