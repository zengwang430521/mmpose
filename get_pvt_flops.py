# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmcv import Config

from mmpose.models import build_posenet

try:
    from mmcv.cnn.utils.flops_counter import get_model_complexity_info, flops_to_string, params_to_string
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')




# W, H = 192, 256
# input_shape = (3, W, H)
#
# # config = 'configs/pvt3h2_den0f_att_adamw.py'
# config = 'configs/pvt3h2_den0f_part0_att_adamw.py'

# config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/den0f_small_adamw_coco_256x192.py'
# config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrtc_bi_part_re17_w32_coco_256x192_scratch.py '
# config = 'configs/pvtv2_0.py'
# config = 'configs/den0f_fine_adamw.py'
# config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/' \
#            'hrtc_bi_part_re17_w32_coco_256x192_scratch.py'
# config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/' \
#            'hrtc_bi_part_re14_w32_coco_256x192_scratch.py'
# cfg = Config.fromfile(config)



W, H = 256, 256
input_shape = (3, W, H)
# config = 'configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/att1_den0f_tiny_wflw_256x256.py'
# config = 'configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/wflw/den0_tiny_wflw_256x256.py'
config = '/home/wzeng/mycodes/mmpose_mine/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/hrtc_re18_w32_mpii_256x256.py'
cfg = Config.fromfile(config)


# W, H = 288, 384
# input_shape = (3, W, H)
# # config = 'configs/debug_den0fs_large_fine0_384x288.py'
# # config = 'configs/den0fs_part_large_fine0_384x288.py'
# config = 'configs/den0fs_part_debug_large_fine0_384x288.py'
# cfg = Config.fromfile(config)
# cfg.model.backbone.pretrained = None

model = build_posenet(cfg.model)
model = model.cuda()
model.eval()

if hasattr(model, 'forward_dummy'):
    model.forward = model.forward_dummy
else:
    raise NotImplementedError(
        'FLOPs counter is currently not currently supported with {}'.
        format(model.__class__.__name__))

flops, params = get_model_complexity_info(model, input_shape, as_strings=False)

back = model.backbone.get_extra_flops(H, W)
flops += back
if model.with_neck:
    neck = model.neck.get_extra_flops(H//4, W//4)
    flops += neck



flops = flops_to_string(flops)
params = params_to_string(params)



split_line = '=' * 30
print(f'{split_line}\nInput shape: {input_shape}\n'
      f'Flops: {flops}\nParams: {params}\n{split_line}')
print('!!!Please be cautious if you use the results in papers. '
      'You may need to check if all ops are supported and verify that the '
      'flops computation is correct.')




from mmcv.cnn.utils.flops_counter import get_model_complexity_info

model.forward = model.forward_dummy

flops1, params1 = get_model_complexity_info(encoder, (3, 2048), as_strings=False)
flops2, params2 = get_model_complexity_info(decoder, (3, 2048), as_strings=False)


#
# import torch.nn as nn
# from mmcv.cnn.utils.flops_counter import get_model_complexity_info
#
# class FakeNet(nn.Module):
#     def __init__(self, E, D):
#         super().__init__()
#         self.E = E
#         self.D = D
#
#     def forward(self, x):
#         y = self.E([x])
#         z = self.D([y])
#         return z
#
# fakenet = FakeNet(self.E, self.D)
# flops2, params2 = get_model_complexity_info(fakenet, (3, 2048), as_strings=False)
# print(flops)
# print(params2)


