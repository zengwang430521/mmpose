#!/usr/bin/env bash
export MASTER_PORT=29505


srun -p mm_human --quotatype=auto\
srun -p pat_earth -x SH-IDC1-10-198-4-[90-91,100-103,116-119] \
srun -p pat_earth -x SH-IDC1-10-198-4-[87,100-103,116-119] \
    --ntasks=8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --job-name=hrtw32_pre_adamw python -u tools/train.py configs/hrtw32_pre_adamw.py --work-dir=work_dirs/hrtw32_pre_adamw --launcher="slurm" \
    --resume-from=work_dirs/hrtw32_pre_adamw/latest.pth


python -m torch.distributed.launch --nproc_per_node=8 --master_port=29876 \
    tools/train.py configs/ablation_att.py --launcher pytorch --work-dir=work_dirs/ablation_att
