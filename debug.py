


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


#####################################################################

import torch
# from mmpose.models.backbones import mypvt3h2_density0fNC_small
# model = mypvt3h2_density0fNC_small(pretrained='data/pretrained/my_20_2_300.pth')

from mmpose.models.backbones import tcformer_hir_small
model = tcformer_hir_small().cuda()

input = torch.rand([1, 3, 192, 256]).cuda()
out = model(input)
out = out




#####################################################################

# import torch
# import mmpose.models.backbones.utils_mine as utils_mine
# device = torch.device('cuda')
# B, N, C, Ns = 2, 1000, 64, 400
# x = torch.rand(B, N, C).to(device)
# conf = torch.rand(B, N, 1).to(device)
# idx_agg = torch.arange(N)[None, :].repeat(B, 1).to(device)
#
# x_down = utils_mine.token_cluster_hir(x, Ns, idx_agg, conf, return_weight=True)
#



#####################################################################






# from mmpose.models import build_posenet




#
#
# channel_cfg = dict(
#     num_output_channels=133,
#     dataset_joints=133,
#     dataset_channel=[
#         list(range(133)),
#     ],
#     inference_channel=list(range(133)))
#
#
# norm_cfg = dict(type='BN', requires_grad=True)
# # model settings
# model = dict(
#     type='TopDown',
#     # pretrained='torchvision://resnet50',
#     # backbone=dict(type='ResNet', depth=50),
#     backbone=dict(type='mypvt3h2_density0f_tiny', pretrained='models/3h2_density0f_tiny.pth'),
#     keypoint_head=dict(
#         type='TopdownHeatmapSimpleHead',
#         # in_channels=2048,
#         in_channels=512,
#         in_index=3,
#         out_channels=channel_cfg['num_output_channels'],
#         loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
#     train_cfg=dict(),
#     test_cfg=dict(
#         flip_test=True,
#         post_process='default',
#         shift_heatmap=True,
#         modulate_kernel=11))
#
#
#
#
# device = torch.device('cuda')
#
#
# model = build_posenet(model).to(device)
# input = torch.rand([2, 3, 256, 256], device=device)
# out = model(input)


# import  torch
# device = torch.device('cuda')
# x = torch.zeros(2, 3136, 128).to(device)
# idx = torch.rand(2, 3136).to(device) * 48
# idx = idx.long()
# idx_batch = torch.arange(2)[:, None].expand(2, 3136).to(device)
# y = torch.zeros(2, 64, 49, 128).to(device)
# # z = y[idx_batch.reshape(-1), idx.reshape(-1), :]
# # z[:, :, :]= 1
#
# out = torch.bmm(y[idx_batch.reshape(-1), idx.reshape(-1), :], x.reshape(-1, 128, 1))
#


# import torch
# from typing import List
#
# def foo(x):
#     return torch.neg(x)
#
# @torch.jit.script
# def example(x):
#     futures : List[torch.jit.Future[torch.Tensor]] = []
#     for _ in range(100):
#         futures.append(torch.jit.fork(foo, x))
#
#     results = []
#     for future in futures:
#         results.append(torch.jit.wait(future))
#
#     return torch.sum(torch.stack(results))
#
# print(example(torch.ones([])))


import torch
from typing import List
#
# @torch.jit.script
# def index_matmul(x, y, idx):
#     B, N, C = x.shape
#     idx_batch = torch.arange(B, device=x.device)
#
#     futures: List[torch.jit.Future[torch.Tensor]] = []
#     for i in range(N):
#         # futures.append(x[:, i, :] @ y[idx_batch, :,  idx[:, i]])
#         futures.append(
#             torch.jit.fork(
#                 torch.matmul,
#                 x[:, i, :].unsqueeze(1), y[idx_batch, :,  idx[:, i]])
#         )
#
#     results = []
#     for future in futures:
#         results.append(torch.jit.wait(future))
#
#     return torch.cat(results, dim=1)
#
#
# def index_matmul_for(x, y, idx):
#     B, N, C = x.shape
#     idx_batch = torch.arange(B, device=x.device)
#
#     futures = []
#     for i in range(N):
#         # futures.append(x[:, i, :] @ y[idx_batch, :,  idx[:, i]])
#         futures.append(
#             torch.matmul(x[:, i, :].unsqueeze(1), y[idx_batch, :,  idx[:, i]])
#         )
#     return torch.cat(futures, dim=1)
#
#
# import  torch
# device = torch.device('cuda')
# x = torch.zeros(2, 3136, 128).to(device)
# idx = torch.rand(2, 3136).to(device) * 48
# idx = idx.long()
# idx_batch = torch.arange(2)[:, None].expand(2, 3136).to(device)
# y = torch.zeros(2, 128, 49, 64).to(device)
#
# import time
# t1 = time.time()
# for n in range(100):
#     out = index_matmul(x, y, idx_batch)
# t2 = time.time()
# print(t2-t1)
#
# t1 = time.time()

# for n in range(100):
#     out = index_matmul(x, y, idx_batch)
# t2 = time.time()
# print(t2-t1)
#
#
#
# t=0

#
# import mmcv
# from mmcv import Config, DictAction
# from mmpose.models import build_posenet
#
#
# # cfg_file = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/debug_myhrpvt32_adamw_coco_256x192.py'
# # cfg_file = 'configs/debug_den0fs_large_fine0_384x288.py'
# cfg_file = 'configs/pvtv2_att_fine_adamw.py'
# cfg = Config.fromfile(cfg_file)
# model = cfg.model
# model = build_posenet(model) #.cuda().half()
# input = torch.zeros(2,3, 256, 192)
# out = model(input)
#
#
#
# import matplotlib.pyplot as plt
# data = torch.load('NAN_debug.pth', map_location='cuda:0')
# img = data['img']
# # img = img[24:26]
# target = data['target']
# target_weight = data['target_weight']
# output = data['output']
# state_dict = data['model']
#
# IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406], device=img.device)[None, :, None, None]
# IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225], device=img.device)[None, :, None, None]
# img_ori = img.float() * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN
# t = img_ori[1].float().permute(1,2,0).detach().cpu()
# t = t.clamp(0, 1)
# plt.imshow(t.numpy())
#
#
#
# model = build_posenet(model) #.cuda().half()
# t = model.load_state_dict(state_dict)
#
#
# img = img.float()
# model = model.cuda()
# model.eval()
# output = model.backbone(img[4:6])
# output = model.neck(output)
# output = model.keypoint_head(output)
#
# losses = dict()
# keypoint_losses = model.keypoint_head.get_loss(output, target, target_weight)
# losses.update(keypoint_losses)
# keypoint_accuracy = model.keypoint_head.get_accuracy(
#     output, target, target_weight)
# losses.update(keypoint_accuracy)
#
#
#
# for key in state_dict.keys():
#     print(key)
#
# tmp = model.state_dict()
#
#
#
#
#
# data = torch.load('NAN_debug.pth', map_location='cuda:0')
# img = data['img']
# # target = data['target']
# # target_weight = data['target_weight']
# # output = data['output']
# state_dict = data['model']
# self.load_state_dict(state_dict)
#
# with torch.no_grad():
#     output1 = self.backbone(img)
#     if self.with_neck:
#         output = self.neck(output)
#     if self.with_keypoint:
#         output2 = self.keypoint_head(output1)
#
#
# for t in x_list:
#     print(torch.isinf(t['x']).any())
#     print(torch.isnan(t['x']).any())
#
# for t in y_list:
#     print(torch.isinf(t['x']).any())
#     print(torch.isnan(t['x']).any())


from mmpose.models.backbones.tc_module.hr_tcformer import HRTCFormer

norm_cfg = dict(type='BN', requires_grad=True)
model_dict = dict(
    in_channels=3,
    norm_cfg=norm_cfg,
    extra=dict(
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
            block='TRANSFORMER_BLOCK',
            num_blocks=(2, 2),
            num_channels=(32, 64),
            num_heads=[1, 2],
            num_mlp_ratios=[4, 4],
            num_window_sizes=[7, 7]),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block='TCWIN_BLOCK',
            num_blocks=(2, 2, 2),
            num_channels=(32, 64, 128),
            num_heads=[1, 2, 4],
            num_mlp_ratios=[4, 4, 4],
            num_window_sizes=[7, 7, 7]),
        stage4=dict(
            num_modules=2,
            num_branches=4,
            block='TCWIN_BLOCK',
            num_blocks=(2, 2, 2, 2),
            num_channels=(32, 64, 128, 256),
            num_heads=[1, 2, 4, 8],
            num_mlp_ratios=[4, 4, 4, 4],
            num_window_sizes=[7, 7, 7, 7])
    )
)

device = torch.device('cuda')
model = HRTCFormer(**model_dict).to(device)

B, H, W, C = 1, 257, 192, 3
img = torch.zeros([B, C, H, W], device=device)

out = model(img)