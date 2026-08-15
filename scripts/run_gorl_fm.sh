nohup env XLA_PYTHON_CLIENT_PREALLOCATE=false \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/root/GoRL:/root/GoRL/envs/robomimic \
python -u scripts/run_gorl_fm.py \
> logs/gorl_origin-$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &

nohup env XLA_PYTHON_CLIENT_PREALLOCATE=false \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/root/GoRL:/root/GoRL/envs/robomimic \
python -u scripts/run_gorl_fm.py \
--no-stage-init-before-training \
> logs/gorl_origin-$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &

nohup env XLA_PYTHON_CLIENT_PREALLOCATE=false \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/root/GoRL:/root/GoRL/envs/robomimic \
python -u scripts/run_gorl_fm.py \
--no-stage-init-before-training \
--use-offline-checkpoint \
--offline-checkpoint-path /root/GoRL/results/offline_fm_frozen_robomimic_20260804_202346/checkpoint_final.pkl \
> logs/gorl_origin-$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &

nohup env XLA_PYTHON_CLIENT_PREALLOCATE=false \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES=3 \
PYTHONPATH=/root/GoRL:/root/GoRL/envs/robomimic \
python -u scripts/run_gorl_fm.py \
--use-offline-checkpoint \
--offline-checkpoint-path /root/GoRL/results/offline_fm_frozen_robomimic_20260804_202346/checkpoint_final.pkl \
> logs/gorl_origin-$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &


nohup env XLA_PYTHON_CLIENT_PREALLOCATE=false \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 \
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH=/root/GoRL:/root/GoRL/envs/robomimic \
python -u run_offline_fm_frozen_robomimic.py \
> logs/gorl_offline-$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
