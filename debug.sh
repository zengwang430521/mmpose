#!/usr/bin/env bash
export MASTER_PORT=29505

srun -p mm_human \
    --ntasks=4 --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --job-name=ae_att_coco python -u tools/train.py  --work-dir=work_dirs/ae_att_coco --launcher="slurm" \
    configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/den0_small_coco_512x512.py


srun -p pat_earth \
srun -p pat_earth -x SH-IDC1-10-198-4-[90,91,100-103,116-119] \
srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
srun -p mm_human --quotatype=auto\
    --ntasks=8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --job-name=ablation python -u tools/train.py  configs/ablation_noconf.py \
    --work-dir=work_dirs/ablation_noconf --launcher="slurm" --resume=work_dirs/ablation_noconf/latest.pth


    --job-name=fine_simple2 python -u tools/train.py  configs/fine_simple.py \
    --work-dir=work_dirs/fine_simple2 --launcher="slurm" --resume=work_dirs/fine_simple/latest.pth


    --job-name=fine_hand0 python -u tools/train.py  configs/fine_hand0.py \
    --work-dir=work_dirs/fine_hand0 --launcher="slurm"

    --job-name=fine_face0 python -u tools/train.py  configs/fine_face0.py \
    --work-dir=work_dirs/fine_face0 --launcher="slurm"




    --job-name=fine_att python -u tools/train.py  configs/fine_att.py \
    --work-dir=work_dirs/fine_att --launcher="slurm" --resume=work_dirs/fine_att/latest.pth




    --job-name=fine_face python -u tools/train.py  configs/fine_face.py \
    --work-dir=work_dirs/fine_face --launcher="slurm"

    --job-name=fine_hand python -u tools/train.py  configs/fine_hand.py \
    --work-dir=work_dirs/fine_hand --launcher="slurm"






    --job-name=ablation python -u tools/train.py  configs/pvtv2_att_fine_adamw.py \
    --work-dir=work_dirs/pvt2_att_whole --launcher="slurm" --resume=work_dirs/pvt2_att_whole/latest.pth

    --job-name=fine_simple python -u tools/train.py  configs/fine_simple.py \
    --work-dir=work_dirs/fine_simple --launcher="slurm" --resume=work_dirs/den0_simple_whole/latest.pth


    --job-name=fine_pvt python -u tools/train.py  configs/fine_pvt2.py \
    --work-dir=work_dirs/fine_pvt --launcher="slurm" --resume=work_dirs/pvtv2/latest.pth


    --job-name=fine_hrformer python -u tools/train.py  configs/fine_hrtw32.py \
    --work-dir=work_dirs/fine_hrformer --launcher="slurm" --resume=work_dirs/hrtw32_pre/latest.pth

    --job-name=fine_att python -u tools/train.py  configs/fine_att.py \
    --work-dir=work_dirs/fine_att --launcher="slurm" --resume=work_dirs/den0f_att_16/latest.pth





    --job-name=fine_att python -u tools/train.py  configs/den0f_att_fine2_adamw.py \
    --work-dir=work_dirs/den0f_att_fine2 --launcher="slurm" --resume=work_dirs/den0f_att_16/latest.pth


    --job-name=myhrpvt32 python -u tools/train.py  configs/den0f_fine_adamw.py \
    --work-dir=work_dirs/den0_simple_whole --launcher="slurm" --resume=work_dirs/den0_simple_whole/latest.pth

    --job-name=myhrpvt32 python -u tools/train.py  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/myhrpvt32_adamw_coco_256x192.py\
    --work-dir=work_dirs/myhrpvt_coco_3 --launcher="slurm" --resume=work_dirs/myhrpvt_coco_3/epoch_26.pth

srun -p mm_human --quotatype=auto\
    --ntasks=8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --job-name=myhrpvt32 python -u tools/train.py  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/myhrpvt32_adamw_coco_256x192.py\
    --work-dir=work_dirs/myhrpvt_coco_2 --launcher="slurm" --resume=work_dirs/myhrpvt_coco/epoch_25.pth

    --job-name=myhrpvt32 python -u tools/train.py  configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/myhrpvt32_adamw_coco_256x192.py\
    --work-dir=work_dirs/myhrpvt_coco --launcher="slurm"

srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
    --ntasks=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=2 --kill-on-bad-exit=1 \
    --job-name=aflw_att1_adamw python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/aflw/att1_adamw_tiny_aflw_256x256.py \
    --work-dir=work_dirs/aflw_att1_adamw --launcher="slurm" --resume=work_dirs/aflw_att1_adamw/latest.pth


    --job-name=aflw_att1fine python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/aflw/att1_fine_den0f_tiny_aflw_256x256.py \
    --work-dir=work_dirs/aflw_att1fine --launcher="slurm" --resume=work_dirs/aflw_att1fine/latest.pth


    --job-name=aflw_att-4 python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/aflw/2e-4att1_den0f_tiny_aflw_256x256.py \
    --work-dir=work_dirs/aflw_att_lr-4 --launcher="slurm" --resume=work_dirs/aflw_att_lr-4/latest.pth

    --job-name=aflw_att1n python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/aflw/att1n_den0f_tiny_aflw_256x256.py \
    --work-dir=work_dirs/aflw_att1n --launcher="slurm" --resume=work_dirs/aflw_att1n/latest.pth

srun -p pat_earth \
    --ntasks=8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
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



srun -p mm_human --quotatype=auto\
srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
srun -p pat_earth \
    --ntasks=16 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --job-name=debug python -u tools/train.py configs/collect_den0fs_large_fine0_384x288.py\
    --work-dir=work_dirs/den0fs_large_384_debug --launcher="slurm"  --resume=work_dirs/den0fs_large_384/latest.pth

    --ntasks=32 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --job-name=l_s_fine_att python -u tools/train.py configs/den0fs_large_fine0_384x288.py\
    --work-dir=work_dirs/den0fs_large_384_32 --launcher="slurm"  --resume=work_dirs/den0fs_large_384/latest.pth


srun -p pat_earth \
srun -p pat_earth -x SH-IDC1-10-198-4-[90,91,100-103,116-119] \
    --ntasks=16 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --job-name=l_s_fine_att python -u tools/train.py configs/den0fs_large_fine0_384x288.py\
     --work-dir=work_dirs/den0fs_large_384_16 --launcher="slurm"  --resume=work_dirs/den0fs_large_384_32/latest.pth








    --job-name=aflw_pvt python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/aflw/pvt_tiny_aflw_256x256.py \
    --work-dir=work_dirs/aflw_pvt --launcher="slurm" --resume=work_dirs/aflw_pvt/latest.pth

    --job-name=wflw_den0 python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/den0_tiny_wflw_256x256.py \
    --work-dir=work_dirs/wflw_den0 --launcher="slurm" --resume=work_dirs/wflw_den0/latest.pth





srun -p pat_earth -x SH-IDC1-10-198-4-[100-103,116-119] \
srun -p pat_earth \
    --ntasks=1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=2 --kill-on-bad-exit=1 \
    --job-name=eval python -u tools/test.py  configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/att1_den0f_tiny_wflw_blur_256x256.py  \
    work_dirs/wflw_att1/epoch_60.pth --launcher="slurm"




    --job-name=eval python -u tools/test.py  configs/debug_den0fs_att_adamw.py work_dirs/den0f_att_16/epoch_210.pth --launcher="slurm"

    --job-name=aflw_den0 python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/aflw/den0_tiny_aflw_256x256.py \
    --work-dir=work_dirs/aflw_den0 --launcher="slurm" --resume=work_dirs/aflw_den0/latest.pth



    --job-name=wflw_pvt python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/pvt_tiny_wflw_256x256.py \
    --work-dir=work_dirs/wflw_pvt --launcher="slurm"


    --job-name=aflw_att1 python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/aflw/att1_den0f_tiny_aflw_256x256.py \
    --work-dir=work_dirs/aflw_att1 --launcher="slurm"

    --job-name=wflw_2e-4_fpn0_den0f python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/2e-4fpn0_den0f_tiny_wflw_256x256.py \
    --work-dir=work_dirs/wflw_2e-4_fpn0_den0f --launcher="slurm"

    --job-name=wflw_2e-4_fpn_base python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/2e-4fpn_pvt_tiny_wflw_256x256.py \
    --work-dir=work_dirs/wflw_2e-4_fpn_base --launcher="slurm"


    --job-name=wflw_att5 python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/att5_den0f_tiny_wflw_256x256.py \
    --work-dir=work_dirs/wflw_att5 --launcher="slurm"

    --job-name=wflw_att5n python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/att5n_den0f_tiny_wflw_256x256.py \
    --work-dir=work_dirs/wflw_att5n --launcher="slurm"

    --job-name=wflw_fpn0_den0f python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/fpn0_den0f_tiny_wflw_256x256.py \
    --work-dir=work_dirs/wflw_fpn0_den0f --launcher="slurm"

    --job-name=wflw_fpn_base python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/fpn_pvt_tiny_wflw_256x256.py \
    --work-dir=work_dirs/wflw_fpn_base --launcher="slurm"


     --job-name=wflw_att1_l2 python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/att1_lr2_den0f_tiny_wflw_256x256.py \
    --work-dir=work_dirs/wflw_att1_l2 --launcher="slurm"

    --job-name=wflw_att4 python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/att4_den0f_tiny_wflw_256x256.py \
    --work-dir=work_dirs/wflw_att4 --launcher="slurm"


    --job-name=wflw_att1_l python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/att1_lr_den0f_tiny_wflw_256x256.py \
    --work-dir=work_dirs/wflw_att1_l --launcher="slurm"

    --job-name=wflw_hr python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/hr_den0f_tiny_wflw_256x256.py \
    --work-dir=work_dirs/wflw_hr --launcher="slurm"

    --job-name=wflw_fpn python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/fpn_den0f_tiny_wflw_256x256.py \
    --work-dir=work_dirs/wflw_fpn --launcher="slurm"

     --job-name=ae_att_coco python -u tools/train.py  --work-dir=work_dirs/ae_att_coco --launcher="slurm" \
    configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/den0fs_small_coco_512x512.py

    --job-name=wflw_att3 python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/att3_den0f_tiny_wflw_256x256.py \
    --work-dir=work_dirs/wflw_att3 --launcher="slurm"

    --job-name=wflw_att2 python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/att2_den0f_tiny_wflw_256x256.py \
    --work-dir=work_dirs/wflw_att2 --launcher="slurm"


    --job-name=wflw_att1 python -u tools/train.py configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/att1_den0f_tiny_wflw_256x256.py \
    --work-dir=work_dirs/wflw_att1 --launcher="slurm"





srun -p mm_human \
srun -p pat_earth -x SH-IDC1-10-198-4-[90-91,100-103,116-119] \
srun -p pat_earth -x SH-IDC1-10-198-4-[87,100-103,116-119] \
    --ntasks=8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --job-name=hrtw32_pre_adamw python -u tools/train.py configs/hrtw32_pre_adamw.py --work-dir=work_dirs/hrtw32_pre_adamw --launcher="slurm" \
    --resume-from=work_dirs/hrtw32_pre_adamw/latest.pth


     --job-name=f_att_coco python -u tools/train.py  --work-dir=work_dirs/att_coco --launcher="slurm" \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/den0f_small_coco_256x192.py --resume-from=work_dirs/att_coco/latest.pth




    --job-name=den0f_att2 python -u tools/train.py configs/den0f_att2_adamw.py --work-dir=work_dirs/den0f_att2_8 --launcher="slurm"

    --job-name=ae_att_coco python -u tools/train.py  --work-dir=work_dirs/ae_att_coco --launcher="slurm" \
    configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/den0fs_small_coco_512x512.py


    --job-name=f_w_att_coco python -u tools/train.py  --work-dir=work_dirs/adamw_att_coco --launcher="slurm" \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/den0f_small_adamw_coco_256x192.py


    --job-name=den0f_att python -u tools/train.py configs/pvt3h2_den0f_att_adamw.py --work-dir=work_dirs/den0f_att_8 --launcher="slurm"

    --job-name=f_att_coco python -u tools/train.py  --work-dir=work_dirs/att_coco --launcher="slurm" \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/den0f_small_coco_256x192.py



    --job-name=den0_att python -u tools/train.py configs/pvt3h2_den0_att_adamw.py --work-dir=work_dirs/den0_att --launcher="slurm"



    --job-name=denc_fpn python -u tools/train.py configs/pvt3h2_denc_fpn_adamw.py --work-dir=work_dirs/denc_fpn_16 --launcher="slurm"



    --job-name=den0_fpn python -u tools/train.py configs/pvt3h2_den0_fpn_adamw.py --work-dir=work_dirs/den0_fpn_16 --launcher="slurm"

    --job-name=den0_catgau python -u tools/train.py configs/pvt3h2_den0_catgau_adamw.py --work-dir=work_dirs/den0_catgau --launcher="slurm"

    --job-name=den0_fpn python -u tools/train.py configs/pvt3h2_den0_fpn_adamw.py --work-dir=work_dirs/den0_fpn --launcher="slurm"

    --job-name=pvt3h2fn_cat_gau python -u tools/train.py configs/pvt3h2fn_cat_gau.py --work-dir=work_dirs/my3h2fn_cat_gau --launcher="slurm"

   --job-name=pvt3h2_dwcat_adamw python -u tools/train.py configs/pvt3h2_dwcat_adamw.py --work-dir=work_dirs/my3h2_dwcat_adamw --launcher="slurm"

   --job-name=pvt3h2_fpn_adamw python -u tools/train.py configs/pvt3h2_fpn_adamw.py --work-dir=work_dirs/my3h2_fpn_adamw --launcher="slurm"

   --job-name=pvt3h2a_fpn_adamw python -u tools/train.py configs/pvt3h2a_fpn_adamw.py --work-dir=work_dirs/my3h2a_fpn_adamw --launcher="slurm"

   --job-name=pvtv2_fpn_pre_adamw python -u tools/train.py configs/pvtv2_fpn_pre_adamw.py --work-dir=work_dirs/pvtv2_fpn_pre_adamw --launcher="slurm"

   --job-name=pvtv2_cat_pre_adamw python -u tools/train.py configs/pvtv2_cat_pre_adamw.py --work-dir=work_dirs/pvtv2_cat_pre_adamw --launcher="slurm"


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




# for demo


python demo/top_down_img_demo.py

configs/debug_den0fs_att_adamw.py work_dirs/fine_att/epoch_250.pth --img-root=data/coco/val2017/
--json-file=data/coco/annotations/coco_wholebody_val_v1.0.json --out-img-root=demo_results



configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/att1_den0f_tiny_wflw_256x256.py
 work_dirs/wflw_att1/epoch_60.pth
 --img-root=data/wflw/images/
--json-file=data/wflw/annotations/face_landmarks_wflw_train.json --out-img-root=demo_results


