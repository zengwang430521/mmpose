import torch
from mmpose.models.backbones.tc_module.hr_tcformer import HRTCFormer
from mmpose.models import build_posenet, build_backbone
import copy

channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

norm_cfg = dict(type='BN', requires_grad=True)

backbone = dict(
    type='HRTCFormer',
    in_channels=3,
    norm_cfg=norm_cfg,
    return_map=True,
    extra=dict(
        nh_list=[8, 4, 2],
        nw_list=[8, 4, 2],
        drop_path_rate=0.1,
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block='BOTTLENECK',
            num_blocks=(2,),
            num_channels=(64,),
            num_heads=[2],
            num_mlp_ratios=[4]),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block='TCWINBLOCK',
            num_blocks=(2, 2),
            num_channels=(32, 64),
            num_heads=[1, 2],
            num_mlp_ratios=[4, 4],
            num_window_sizes=[7, 7]),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block='TCWINBLOCK',
            num_blocks=(2, 2, 2),
            num_channels=(32, 64, 128),
            num_heads=[1, 2, 4],
            num_mlp_ratios=[4, 4, 4],
            num_window_sizes=[7, 7, 7]),
        stage4=dict(
            num_modules=2,
            num_branches=4,
            block='TCWINBLOCK',
            num_blocks=(2, 2, 2, 2),
            num_channels=(32, 64, 128, 256),
            num_heads=[1, 2, 4, 8],
            num_mlp_ratios=[4, 4, 4, 4],
            num_window_sizes=[7, 7, 7, 7],
            multiscale_output=True),

))

device = torch.device('cuda')

model = build_backbone(backbone).to(device)

x = torch.rand([1, 3, 224, 224]).to(device)

y = model(x)

l = y[1].sum()
l.backward()

