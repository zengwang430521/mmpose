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

# channel_cfg = dict(
#     num_output_channels=133,
#     dataset_joints=133,
#     dataset_channel=[
#         list(range(133)),
#     ],
#     inference_channel=list(range(133)))
#
# # norm_cfg = dict(type='SyncBN', requires_grad=True)
# norm_cfg = dict(type='BN', requires_grad=True)
# data_cfg = dict(
#     image_size=512,
#     base_size=256,
#     base_sigma=2,
#     heatmap_size=[128, 256],
#     num_joints=channel_cfg['dataset_joints'],
#     dataset_channel=channel_cfg['dataset_channel'],
#     inference_channel=channel_cfg['inference_channel'],
#     num_scales=2,
#     scale_aware_sigma=False,
# )
# model = dict(
#     type='AssociativeEmbedding',
#     backbone=dict(type='mypvt3h2_density0_small', pretrained='models/3h2_density0_small.pth',),
#     neck=dict(
#         type='AttenNeck',
#         in_channels=[64, 128, 320, 512],
#         out_channels=128,
#         start_level=0,
#         # add_extra_convs='on_input',
#         num_outs=1,
#         num_heads=[4, 4, 4, 4],
#         mlp_ratios=[4, 4, 4, 4],
#     ),
#     keypoint_head=dict(
#         type='AEHigherResolutionHead',
#         in_channels=128,
#         num_joints=17,
#         tag_per_joint=True,
#         extra=dict(final_conv_kernel=1, ),
#         num_deconv_layers=1,
#         num_deconv_filters=[128],
#         num_deconv_kernels=[4],
#         num_basic_blocks=4,
#         cat_output=[True],
#         with_ae_loss=[True, False],
#         loss_keypoint=dict(
#             type='MultiLossFactory',
#             num_joints=17,
#             num_stages=2,
#             ae_loss_type='exp',
#             with_ae_loss=[True, False],
#             push_loss_factor=[0.001, 0.001],
#             pull_loss_factor=[0.001, 0.001],
#             with_heatmaps_loss=[True, True],
#             heatmaps_loss_factor=[1.0, 1.0])),
#     train_cfg=dict(
#         num_joints=channel_cfg['dataset_joints'],
#         img_size=data_cfg['image_size']),
#     test_cfg=dict(
#         num_joints=channel_cfg['dataset_joints'],
#         max_num_people=30,
#         scale_factor=[1],
#         with_heatmaps=[True, True],
#         with_ae=[True, False],
#         project2image=True,
#         nms_kernel=5,
#         nms_padding=2,
#         tag_per_joint=True,
#         detection_threshold=0.1,
#         tag_threshold=1,
#         use_detection_val=True,
#         ignore_too_much=False,
#         adjust=True,
#         refine=True,
#         flip_test=True))
#
# device = torch.device('cuda')
# model = build_posenet(model).to(device)
# input = torch.rand([2, 3, 192, 256], device=device)
# out = model(input)
# out = out
#


import matplotlib
import matplotlib.pyplot as plt
import cv2

fname = 'vis/1272_heatmap.pth'
data = torch.load(fname)
x = data['x']
_, _, h, w = x.shape
x = torch.nn.functional.interpolate(x, [h*4, w*4], mode='nearest')
for i in range(133):
    heatmap = x[0, i].cpu()
    heatmap = torch.cat([heatmap[[0], :], heatmap], dim=0)
    # heatmap = heatmap - heatmap.min()
    # heatmap = heatmap /heatmap.max()
    # cmap = matplotlib.cm.get_cmap("viridis", 8)
    cmap = matplotlib.cm.get_cmap()
    h = cmap(heatmap)
    h = h[1:, ]
    plt.imshow(h[:, :, :])
    cv2.imwrite(f'heatmap_{i}.png', h[:, :, 2::-1]*255)





from mmpose.models.backbones.utils_mine import vis_tokens_merge
vis_tokens_merge(563)