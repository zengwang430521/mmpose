import torch

src = 'work_dirs/den0f_att_16/epoch_210.pth'
dst = 'work_dirs/coco_att1_neck.pth'
src_dict = torch.load(src)
src_dict = src_dict['state_dict']
tar_dict = {}
for key in src_dict.keys():
    if 'neck' in key:
        tar_key = key.replace('neck.', '')
        tar_dict[tar_key] = src_dict[key]
torch.save(tar_dict, dst)
