# from tests.test_model.test_mesh_forward import test_parametric_mesh_forward
# from tests.test_model.test_mesh_head import test_mesh_hmr_head
#
# test_parametric_mesh_forward()
# test_mesh_hmr_head()
# print('finish')

# from tests.test_datasets.test_hand_dataset import test_top_down_InterHand3D_dataset
# test_top_down_InterHand3D_dataset()

# from tests.test_apis.test_inference import test_bottom_up_demo, test_process_mmdet_results, test_top_down_demo
# test_bottom_up_demo()
# test_process_mmdet_results()
# test_top_down_demo()


import torch

# from mmpose.models.backbones import mypvt20_2_small
# model = mypvt20_2_small(pretrained='data/pretrained/my_20_2_300.pth')
# input = torch.rand([1, 3, 192, 256])
# out = model(input)
# out = out


from mmpose.models import build_posenet






channel_cfg = dict(
    num_output_channels=133,
    dataset_joints=133,
    dataset_channel=[
        list(range(133)),
    ],
    inference_channel=list(range(133)))


norm_cfg = dict(type='BN', requires_grad=True)
# model settings
model = dict(
    type='TopDown',
    # pretrained='/path/to/hrt_small.pth', # Set the path to pretrained backbone here
    backbone=dict(
        type='MyHRPVT',
        in_channels=3,
        norm_cfg=norm_cfg,
        return_map=True,
        extra=dict(
            drop_path_rate=0.1,
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(2, ),
                num_channels=(64, ),
                num_heads=[2],
                num_mlp_ratios=[4]),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                remerge=(False, False),
                block='MYBLOCK',
                num_blocks=(2, 2),
                num_channels=(32, 64),
                num_heads=[1, 2],
                num_mlp_ratios=[4, 4],
                sr_ratios=[8, 4]),
            stage3=dict(
                num_modules=4,
                remerge=(False, False, False, False),
                num_branches=3,
                block='MYBLOCK',
                num_blocks=(2, 2, 2),
                num_channels=(32, 64, 128),
                num_heads = [1, 2, 4],
                num_mlp_ratios = [4, 4, 4],
                sr_ratios=[8, 4, 2]),
            stage4=dict(
                num_modules=2,
                remerge=(False, False),
                num_branches=4,
                block='MYBLOCK',
                num_blocks=(2, 2, 2, 2),
                num_channels=(32, 64, 128, 256),
                num_heads = [1, 2, 4, 8],
                num_mlp_ratios = [4, 4, 4, 4],
                sr_ratios=[8, 4, 2, 1])
            )),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=32,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        # norm_cfg=norm_cfg,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))


device = torch.device('cuda')


model = build_posenet(model).to(device)
input = torch.rand([2, 3, 256, 256], device=device)
out = model(input)


