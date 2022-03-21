srun -p pat_earth \
    --ntasks=8 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --kill-on-bad-exit=1 \
    --job-name=eval python -u tools/test.py   \
    --launcher="slurm"

    configs/body/3d_mesh_sview_rgb_img/tcformer_hir2_mixed_224x224.py

    models/pytorch/hmr/hmr_mesh_224x224-c21e8229_20201015.pth 8 --eval=joint_error
