from dataclasses import asdict, dataclass


@dataclass
class EnvConfig:
    """Minari dataset and Gymnasium v5 task settings."""
    dataset_path: str = "mujoco/halfcheetah/medium-v0"
    action_repeat: int = 1
    episode_length: int = 1000
    num_envs: int = 48
    eval_num_envs: int = 10#48
    # Evaluate one published policy every N versions.
    eval_interval: int = 10
    dense_reward: bool = False
    success_reward_bonus: float = 0.0
    terminate_on_success: bool = False
    # Whether to keep the decoder fixed during online training.
    freeze_decoder: bool = True

    def to_dict(self):
        return asdict(self)
