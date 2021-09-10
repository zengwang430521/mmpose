#!/usr/bin/env bash


srun -p pat_earth \
    -x SH-IDC1-10-198-4-[100-103,116-119] \
    --job-name=res50 --ntasks=8 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    python -u tools/train.py configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/res50_coco_wholebody_256x192.py \
    --work-dir=work_dirs/res50 --launcher="slurm"
