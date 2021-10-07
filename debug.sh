#!/usr/bin/env bash

srun -p mm_human --ntasks=8 \
    --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --job-name=pvtv2   python -u tools/train.py configs/pvtv2_coco_wholebody_256x192.py \
    --work-dir=work_dirs/pvtv2 --resume-from=work_dirs/pvtv2/latest.pth --launcher="slurm"

srun -p mm_human \
srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
srun -p pat_earth \
    --ntasks=8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --job-name=pvt3h1_up python -u tools/train.py configs/pvt3h1_up.py --work-dir=work_dirs/my3h1_up --launcher="slurm"

    --job-name=pvt3h2_up python -u tools/train.py configs/pvt3h2_up.py --work-dir=work_dirs/my3h2_up --launcher="slurm"

    --job-name=pvt3h2_cat python -u tools/train.py configs/pvt3h2_cat.py --work-dir=work_dirs/my3h2_cat --launcher="slurm"

    --job-name=pvtv2_cat python -u tools/train.py configs/pvtv2_cat.py --work-dir=work_dirs/pvtv2_cat --launcher="slurm"

    --job-name=pvt3g1_up python -u tools/train.py configs/pvt3g1_up.py --work-dir=work_dirs/my3g1_up --launcher="slurm"

    --job-name=pvt3g_up python -u tools/train.py configs/pvt3g_up.py --work-dir=work_dirs/my3g_up --launcher="slurm"

    --job-name=pvt3f12_2_up python -u tools/train.py configs/pvt3f12_2_up.py --work-dir=work_dirs/my3f12_2_up --launcher="slurm"

    --job-name=pvt3f12_1_up python -u tools/train.py configs/pvt3f12_1_up.py --work-dir=work_dirs/my3f12_1new_up --launcher="slurm"

    --job-name=pvtv2_up python -u tools/train.py configs/pvtv2_up.py --work-dir=work_dirs/pvtv2_up --launcher="slurm"  --resume-from=work_dirs/pvtv2_up/latest.pth

    --job-name=pvt5f_up python -u tools/train.py configs/pvt5f_up.py --work-dir=work_dirs/my5f_up --launcher="slurm"

    --job-name=pvtv2_fpn python -u tools/train.py configs/pvtv2_fpn.py --work-dir=work_dirs/pvtv2_fpn --launcher="slurm"

    --job-name=pvt3f12_1_fpn python -u tools/train.py configs/pvt3f12_1_fpn.py --work-dir=work_dirs/my3f12_1_fpn --launcher="slurm"

    --job-name=pvt3f12_1_up python -u tools/train.py configs/pvt3f12_1_up.py --work-dir=work_dirs/my3f12_1_up --launcher="slurm"

    --job-name=res_up python -u tools/train.py configs/res50_up.py --work-dir=work_dirs/res50_up --launcher="slurm"


    --job-name=my20_2_0 python -u tools/train.py \
    configs/my20_2_0.py --work-dir=work_dirs/my20_2_0  --resume-from=work_dirs/my20_2_0/latest.pth --launcher="slurm"

    --job-name=pvt_0 python -u tools/train.py \
    configs/pvtv2_0.py --work-dir=work_dirs/pvtv2_0  --resume-from=work_dirs/pvtv2_0/latest.pth --launcher="slurm"

    --job-name=res50_0 python -u tools/train.py \
    configs/res50_0.py --work-dir=work_dirs/res50_0  --resume-from=work_dirs/res50_0/latest.pth --launcher="slurm"

    --job-name=my20_0 python -u tools/train.py configs/my20_2_0.py \
    --work-dir=work_dirs/my20_2_0 --resume-from=work_dirs/my20_2_0/latest.pth --launcher="slurm"

    --job-name=pvtv2_0   python -u tools/train.py configs/pvtv2_0.py \
    --work-dir=work_dirs/pvtv2_0 --resume-from=work_dirs/pvtv2_0/latest.pth --launcher="slurm"


    --job-name=res50 python -u tools/train.py \
    configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/res50_coco_wholebody_256x192.py \
    --work-dir=work_dirs/res50  --resume-from=work_dirs/res50/latest.pth --launcher="slurm"

    --job-name=my20 \
    python -u tools/train.py configs/my20_2_coco_wholebody_256x192.py --work-dir=work_dirs/my20_2 --launcher="slurm"

    --job-name=pvtv2   python -u tools/train.py configs/pvtv2_coco_wholebody_256x192.py \
    --work-dir=work_dirs/pvtv2 --resume-from=work_dirs/pvtv2/latest --launcher="slurm"




