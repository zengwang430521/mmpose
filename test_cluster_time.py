import copy

import torch
from mmcv import Config
from mmpose.models import build_posenet


cfg_file = 'configs/pvt3h2_den0f_part0_att_adamw.py'
# cfg_file = 'configs/pvt3h2_den0f_att_adamw.py'

cfg = Config.fromfile(cfg_file)
model = cfg.model
model = build_posenet(model)
device = torch.device('cuda')
model = model.to(device)
model.eval()


# x = torch.rand([1, 3, 256, 192], device=device)
# x = torch.ones([1, 3, 224, 224], device=device)
import cv2
x = cv2.imread('/home/wzeng/mycodes/mmpose_mine/tests/data/coco/000000000785.jpg')
x = cv2.resize(x, [224, 224])
x = torch.tensor(x).float().to(device).permute(2, 0, 1)[None, ...]
x = x / 255.0 - 0.5



for i in range(5):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    y = model.backbone(x)
    if model.with_neck:
        y = model.neck(y)
    z = model.keypoint_head(y)

    end.record()
    torch.cuda.synchronize()
    print('ALL time: ')
    print(start.elapsed_time(end))

# with torch.autograd.profiler.profile(enabled=True) as prof:
#     y = model.backbone(x)
#     if model.with_neck:
#         y = model.neck(y)
#     z = model.keypoint_head(y)
# print(prof.key_averages().table(sort_by="self_cpu_time_total"))
#


# (7.630847930908203+1.8012160062789917+1.1321920156478882)/66.30912017822266

# (3.0709760189056396 + 1.3393919467926025+1.2759040594100952) / 60.20608139038086