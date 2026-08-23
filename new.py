import pickle
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

encoder_path = pathlib.Path("/root/GoRL/results/offline_fm_frozen_robomimic_20260821_184814/checkpoint_final.pkl")
with encoder_path.open("rb") as file:
    encoder_checkpoint = pickle.load(file)
# Offline combined checkpoints use ``config`` for the FM decoder and keep
# the encoder config separately. Online encoder checkpoints use ``config``.
encoder_config = encoder_checkpoint.get(
    "rlpd_encoder_config", encoder_checkpoint["config"]
)
print(encoder_config.apply_tanh_in_rollout)
