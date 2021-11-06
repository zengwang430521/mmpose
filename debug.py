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
    num_output_channels=98,
    dataset_joints=98,
    dataset_channel=[
        list(range(98)),
    ],
    inference_channel=list(range(98)))

model = dict(
    type='TopDown',
    backbone=dict(type='mypvt3h2_density0f_tiny', pretrained='models/3h2_density0f_tiny.pth'),
    neck=dict(
        type='AttenNeck5',
        in_channels=[64, 128, 320, 512],
        out_channels=128,
        start_level=0,
        # add_extra_convs='on_input',
        num_outs=1,
        num_heads=[2, 2, 2, 2],
        mlp_ratios=[4, 4, 4, 4],
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=128,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
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


