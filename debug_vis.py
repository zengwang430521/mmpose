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
# # model settings
# norm_cfg = dict(type='BN', requires_grad=True)
#
#
# model = dict(
#     type='TopDown',
#     backbone=dict(type='mypvt3h2_density0f_tiny', pretrained='models/3h2_density0f_tiny.pth'),
#     neck=dict(
#         type='AttenNeck4',
#         in_channels=[64, 128, 320, 512],
#         out_channels=128,
#         start_level=0,
#         # add_extra_convs='on_input',
#         num_outs=1,
#         num_heads=[4, 4, 4, 4],
#         mlp_ratios=[2, 2, 2, 2],
#     ),
#
#     keypoint_head=dict(
#         type='TopdownHeatmapSimpleHead',
#         in_channels=128,
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
#
# device = torch.device('cpu')
#
#
# model = build_posenet(model).to(device)
# input = torch.rand([2, 3, 256, 256], device=device)
# out = model(input)




# import matplotlib
# import matplotlib.pyplot as plt
# import cv2
#
# fname = 'vis/1272_heatmap.pth'
# data = torch.load(fname)
# x = data['x']
# _, _, h, w = x.shape
# x = torch.nn.functional.interpolate(x, [h*4, w*4], mode='nearest')
# for i in range(133):
#     heatmap = x[0, i].cpu()
#     heatmap = torch.cat([heatmap[[0], :], heatmap], dim=0)
#     # heatmap = heatmap - heatmap.min()
#     # heatmap = heatmap /heatmap.max()
#     # cmap = matplotlib.cm.get_cmap("viridis", 8)
#     cmap = matplotlib.cm.get_cmap()
#     h = cmap(heatmap)
#     h = h[1:, ]
#     plt.imshow(h[:, :, :])
#     cv2.imwrite(f'heatmap_{i}.png', h[:, :, 2::-1]*255)
#
#
#
#
#
from mmpose.models.backbones import utils_mine
# utils_mine.vis_feature_map()
# utils_mine.analysis()
utils_mine.vis_tokens_and_grid()
