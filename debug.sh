#!/usr/bin/env bash

    --job-name=hrpvtw32_bn python -u tools/test.py  configs/hrpvtw32_bn.py  work_dirs/hrpvtw32_bn/latest.pth --launcher="slurm"

srun -p mm_human \
    --ntasks=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=5 --kill-on-bad-exit=1 \

srun -p mm_human \
srun -p pat_earth -x SH-IDC1-10-198-4-[90-91,100-103,116-119] \
    --ntasks=8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --job-name=denc_att python -u tools/train.py configs/pvt3h2_denc_att_adamw.py --work-dir=work_dirs/denc_att --launcher="slurm"

    --job-name=denc_fpn python -u tools/train.py configs/pvt3h2_denc_fpn_adamw.py --work-dir=work_dirs/denc_fpn_16 --launcher="slurm"


    --job-name=den0_att python -u tools/train.py configs/pvt3h2_den0_att_adamw.py --work-dir=work_dirs/den0_att --launcher="slurm"

    --job-name=den0_fpn python -u tools/train.py configs/pvt3h2_den0_fpn_adamw.py --work-dir=work_dirs/den0_fpn_16 --launcher="slurm"

    --job-name=den0_catgau python -u tools/train.py configs/pvt3h2_den0_catgau_adamw.py --work-dir=work_dirs/den0_catgau --launcher="slurm"

    --job-name=den0_fpn python -u tools/train.py configs/pvt3h2_den0_fpn_adamw.py --work-dir=work_dirs/den0_fpn --launcher="slurm"

    --job-name=pvt3h2fn_cat_gau python -u tools/train.py configs/pvt3h2fn_cat_gau.py --work-dir=work_dirs/my3h2fn_cat_gau --launcher="slurm"

   --job-name=pvt3h2_dwcat_adamw python -u tools/train.py configs/pvt3h2_dwcat_adamw.py --work-dir=work_dirs/my3h2_dwcat_adamw --launcher="slurm"

   --job-name=pvt3h2_fpn_adamw python -u tools/train.py configs/pvt3h2_fpn_adamw.py --work-dir=work_dirs/my3h2_fpn_adamw --launcher="slurm"

   --job-name=pvt3h2a_fpn_adamw python -u tools/train.py configs/pvt3h2a_fpn_adamw.py --work-dir=work_dirs/my3h2a_fpn_adamw --launcher="slurm"

   --job-name=pvtv2_fpn_pre_adamw python -u tools/train.py configs/pvtv2_fpn_pre_adamw.py --work-dir=work_dirs/pvtv2_fpn_pre_adamw --launcher="slurm"

   --job-name=pvtv2_cat_pre_adamw python -u tools/train.py configs/pvtv2_cat_pre_adamw.py --work-dir=work_dirs/pvtv2_cat_pre_adamw --launcher="slurm"

    --job-name=hrtw32_pre_adamw python -u tools/train.py configs/hrtw32_pre_adamw.py --work-dir=work_dirs/hrtw32_pre_adamw --launcher="slurm"

     --job-name=pvtv2_cat_pre python -u tools/train.py configs/pvtv2_cat_pre.py --work-dir=work_dirs/pvtv2_cat_pre --launcher="slurm"

    --job-name=pvt3h2_cat_gau python -u tools/train.py configs/pvt3h2_cat_gau.py --work-dir=work_dirs/my3h2_cat_gau --launcher="slurm"

    --job-name=myw32_bn python -u tools/train.py configs/myw32_bn.py --work-dir=work_dirs/myw32_bn --launcher="slurm"

    --job-name=hrpvtw32_bn python -u tools/train.py configs/hrpvtw32_bn.py --work-dir=work_dirs/hrpvtw32_bn --launcher="slurm"

    --job-name=pvt3h2a_up2 python -u tools/train.py configs/pvt3h2a_up2.py --work-dir=work_dirs/my3h2a_up2 --launcher="slurm"

    --job-name=hrpvtw32_dwbn python -u tools/train.py configs/hrpvtw32_dwbn.py --work-dir=work_dirs/hrpvtw32_dwbn --launcher="slurm"

    --job-name=hrtw32_pre python -u tools/train.py configs/hrtw32_pre.py --work-dir=work_dirs/hrtw32_pre --launcher="slurm"

    --job-name=hrpvtw32_gn python -u tools/train.py configs/hrpvtw32_gn.py --work-dir=work_dirs/hrpvtw32_gn --launcher="slurm"

    --job-name=hrtw32 python -u tools/train.py configs/hrtw32.py --work-dir=work_dirs/hrtw32 --launcher="slurm"

    --job-name=hrpvtw32 python -u tools/train.py configs/hrpvtw32.py --work-dir=work_dirs/hrpvtw32 --launcher="slurm"

    --job-name=pvt3h11_up python -u tools/train.py configs/pvt3h11_up.py --work-dir=work_dirs/my3h11_up --launcher="slurm"  --resume-from=work_dirs/my3h11_up/latest.pth

    --job-name=pvt3h2_up2 python -u tools/train.py configs/pvt3h2_up2.py --work-dir=work_dirs/my3h2_up2 --launcher="slurm" --resume-from=work_dirs/my3h2_up2/latest.pth

    --job-name=pvtv2_up_nearest python -u tools/train.py configs/pvtv2_up_nearest.py --work-dir=work_dirs/pvtv2_up_nearest --launcher="slurm"

    --job-name=pvtv2_up_nearest python -u tools/train.py configs/pvtv2_up_nearest.py --work-dir=work_dirs/pvtv2_up_nearest --launcher="slurm"   --resume-from=work_dirs/pvtv2_up_nearest/latest.pth

    --job-name=pvtv2_cat_nearest python -u tools/train.py configs/pvtv2_cat_nearest.py --work-dir=work_dirs/pvtv2_cat_nearest --launcher="slurm"

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




