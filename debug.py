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

# model = dict(
#     type='TopDown',
#     # pretrained='torchvision://resnet50',
#     backbone=dict(type='ResNet', depth=50),
#     neck=dict(type='UpSampleNeck', scale_factor=8, mode='bilinear'),
#     keypoint_head=dict(
#         type='TestSimpleHead',
#         in_channels=2048,
#         out_channels=channel_cfg['num_output_channels'],
#         num_deconv_kernels=(3, 3, 3),
#         loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
#     train_cfg=dict(),
#     test_cfg=dict(
#         flip_test=True,
#         post_process='default',
#         shift_heatmap=True,
#         modulate_kernel=11))

# model = dict(
#     type='TopDown',
#     backbone=dict(type='mypvt3f12_1_small', pretrained=None),
#     neck=dict(type='TokenInterNeck2', scale_factor=8),
#     keypoint_head=dict(
#         type='TestSimpleHead',
#         in_channels=512,
#         out_channels=channel_cfg['num_output_channels'],
#         num_deconv_kernels=(3, 3, 3),
#         loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
#     train_cfg=dict(),
#     test_cfg=dict(
#         flip_test=True,
#         post_process='default',
#         shift_heatmap=True,
#         modulate_kernel=11))


# model = dict(
#     type='TopDown',
#     backbone=dict(type='mypvt3f12_1_small', pretrained=None),
#     neck=dict(
#         type='TokenFPN',
#         in_channels=[64, 128, 320, 512],
#         out_channels=256,
#         start_level=0,
#         add_extra_convs='on_input',
#         num_outs=4),
#     keypoint_head=dict(
#         type='TopdownHeatmapSimpleHead',
#         input_transform='resize_concat',
#         in_channels=(256, 256, 256, 256),
#         in_index=(0, 1, 2, 3),
#         out_channels=channel_cfg['num_output_channels'],
#         num_deconv_layers=0,
#         extra=dict(final_conv_kernel=1, ),
#         loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
#     train_cfg=dict(),
#     test_cfg=dict(
#         flip_test=True,
#         post_process='default',
#         shift_heatmap=True,
#         modulate_kernel=11))


# model = dict(
#     type='TopDown',
#     pretrained='https://download.openmmlab.com/mmpose/'
#     'pretrain_models/hrnet_w32-36af842e.pth',
#     backbone=dict(
#         type='HRNet',
#         in_channels=3,
#         extra=dict(
#             stage1=dict(
#                 num_modules=1,
#                 num_branches=1,
#                 block='BOTTLENECK',
#                 num_blocks=(4, ),
#                 num_channels=(64, )),
#             stage2=dict(
#                 num_modules=1,
#                 num_branches=2,
#                 block='BASIC',
#                 num_blocks=(4, 4),
#                 num_channels=(32, 64)),
#             stage3=dict(
#                 num_modules=4,
#                 num_branches=3,
#                 block='BASIC',
#                 num_blocks=(4, 4, 4),
#                 num_channels=(32, 64, 128)),
#             stage4=dict(
#                 num_modules=3,
#                 num_branches=4,
#                 block='BASIC',
#                 num_blocks=(4, 4, 4, 4),
#                 num_channels=(32, 64, 128, 256))),
#     ),
#     keypoint_head=dict(
#         type='TopdownHeatmapSimpleHead',
#         in_channels=32,
#         out_channels=channel_cfg['num_output_channels'],
#         num_deconv_layers=0,
#         extra=dict(final_conv_kernel=1, ),
#         loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
#     train_cfg=dict(),
#     test_cfg=dict(
#         flip_test=True,
#         post_process='default',
#         shift_heatmap=True,
#         modulate_kernel=11))



# model = dict(
#     type='TopDown',
#     backbone=dict(type='pvt_v2_b2', pretrained=None),
#     neck=dict(
#         type='FPN',
#         in_channels=[64, 128, 320, 512],
#         out_channels=256,
#         start_level=0,
#         add_extra_convs='on_input',
#         num_outs=4),
#     keypoint_head=dict(
#         type='TopdownHeatmapSimpleHead',
#         input_transform='resize_concat',
#         in_channels=(256, 256, 256, 256),
#         in_index=(0, 1, 2, 3),
#         out_channels=channel_cfg['num_output_channels'],
#         num_deconv_layers=0,
#         extra=dict(final_conv_kernel=1, ),
#         loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
#     train_cfg=dict(),
#     test_cfg=dict(
#         flip_test=True,
#         post_process='default',
#         shift_heatmap=True,
#         modulate_kernel=11))

#
# model = dict(
#     type='TopDown',
#     backbone=dict(type='pvt_v2_b2', pretrained=None),
#     neck=dict(
#         type='FPN',
#         in_channels=[64, 128, 320, 512],
#         out_channels=256,
#         start_level=0,
#         add_extra_convs='on_input',
#         num_outs=4,
#         resize_add=True,
#     ),
#     keypoint_head=dict(
#         type='TopdownHeatmapSimpleHead',
#         in_channels=256,
#         out_channels=channel_cfg['num_output_channels'],
#         num_deconv_layers=0,
#         extra=dict(final_conv_kernel=1, ),
#         loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
#     train_cfg=dict(),
#     test_cfg=dict(
#         flip_test=True,
#         post_process='default',
#         shift_heatmap=True,
#         modulate_kernel=11))




# model = dict(
#     type='TopDown',
#     backbone=dict(type='mypvt3h2_small', pretrained=None),
#     keypoint_head=dict(
#         type='TestSimpleHead',
#         input_transform='resize_concat',
#         in_channels=(64, 128, 320, 512),
#         in_index=(0, 1, 2, 3),
#         out_channels=channel_cfg['num_output_channels'],
#         num_deconv_layers=0,
#         extra=dict(final_conv_kernel=1, num_conv_layers=1, num_conv_kernels=(1,)),
#         loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
#     train_cfg=dict(),
#     test_cfg=dict(
#         flip_test=True,
#         post_process='default',
#         shift_heatmap=True,
#         modulate_kernel=11))


model = dict(
    type='TopDown',
    backbone=dict(type='pvt_v2_b2', pretrained=None),
    keypoint_head=dict(
        type='TestSimpleHead',
        input_transform='resize_concat',
        in_channels=(64, 128, 320, 512),
        in_index=(0, 1, 2, 3),
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, num_conv_layers=1, num_conv_kernels=(1,)),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

model = build_posenet(model)
input = torch.rand([2, 3, 192, 256])
out = model(input)
out = out