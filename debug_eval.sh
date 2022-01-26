#!/usr/bin/env bash
srun -p pat_earth \
srun -p mm_human --quotatype=auto\
srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
    --ntasks=8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --job-name=eval python -u tools/test.py  --launcher="slurm" \
    configs/den0fs_part_large_fine0_384x288.py work_dirs/den0fs_large_384_16/epoch_210.pth --launcher="slurm"


    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrtc_bi_part_re10_w32_coco_256x192_scratch.py \
    work_dirs/coco/hrtc_bi_part_re10_32/epoch_40.pth

    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrtc_bi_part_re9_w32_coco_256x192_scratch.py \
    work_dirs/coco/hrtc_bi_part_re9_32/epoch_50.pth

    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrtcformer_bi_part_w32_coco_256x192_scratch.py \
    work_dirs/coco/hrtc_bi_part_32/epoch_210.pth

    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrtc_bi_part_re8_w32_coco_256x192_scratch.py \
    work_dirs/coco/hrtc_bi_part_re8_32_scratch/epoch_100.pth

    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrtc_bi_part_re7_w32_coco_256x192_fine.py \
    work_dirs/coco/hrtc_bi_part_32/epoch_210.pth

    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrtc_bi_part_re7_w32_coco_256x192_fine.py \
    work_dirs/coco/hrtc_bi_part_re7_32_fine/epoch_240.pth

    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrtcformer_bi_w32_coco_256x192_scratch2.py \
    work_dirs/coco/hrtc_bi_w32_scratch2/epoch_50.pth


    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrtc_bi_part_re2_w32_coco_256x192_scratch.py \
    work_dirs/coco/hrtc_bi_part_re2_32/epoch_85.pth

    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrtcformer_bi_w32_coco_256x192_scratch2.py \
    work_dirs/coco/hrtc_bi_w32_scratch2/epoch_10.pth

    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrtcformer_bi_w32_coco_256x192.py \
    work_dirs/coco/hrtc_bi_32_scratch/epoch_100.pth

    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrtcformer_bi_w32_coco_256x192.py \
    models/hrtcformer_small_coco_256x192.pth

    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrformer_small_coco_256x192_debug.py \
    models/hrformer_small_coco_256x192.pth






    --job-name=eval python -u tools/test.py   configs/fine_pvt2.py work_dirs/fine_pvt/latest.pth --launcher="slurm"

    --job-name=eval python -u tools/test.py  configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/res50_coco_wholebody_256x192.py\
     ../mmpose_mine/work_dirs/res50_coco_wholebody_256x192-9e37ed88_20201004.pth --launcher="slurm"

     work_dirs/res50_coco_wholebody_256x192-9e37ed88_20201004.pth --launcher="slurm"

     ../mmpose_mine/work_dirs/res50/latest.pth --launcher="slurm"


    --job-name=eval python -u tools/test.py  configs/res50_0.py work_dirs/res50/latest.pth --launcher="slurm"

    --job-name=eval python -u tools/test.py  configs/hrtw32_pre.py work_dirs/hrtw32_pre/latest.pth --launcher="slurm"

    --job-name=eval python -u tools/test.py  configs/pvtv2_coco_wholebody_256x192.py  work_dirs/pvtv2/latest.pth --launcher="slurm"

    --job-name=eval python -u tools/test.py  configs/pvt3h2_den0f_att_adamw.py  work_dirs/den0f_att_16/latest.pth --launcher="slurm"

    --job-name=eval python -u tools/test.py  configs/den0fs_large_fine0_384x288.py work_dirs/den0fs_large_384/latest.pth --launcher="slurm"


srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
srun -p pat_earth \
    --ntasks=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=2 --kill-on-bad-exit=1 \
    --job-name=eval python -u tools/test.py  configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/att1_den0f_tiny_wflw_blur_256x256.py  \
    work_dirs/wflw_att1/epoch_60.pth --launcher="slurm"

    --job-name=eval python -u tools/test.py  configs/debug_den0fs_att_adamw.py work_dirs/den0f_att_16/epoch_210.pth --launcher="slurm"

