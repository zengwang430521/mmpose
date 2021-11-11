import torch

# src = 'data/pretrained/my20_2_300.pth'
# dst = 'data/pretrained/my20_2_300.pth'
# data_dict = torch.load(src)
# if 'state_dict' not in data_dict.keys():
#     data_dict['state_dict'] = data_dict['model']
# torch.save(data_dict, dst)


src = 'work_dirs/den0f_att_16/epoch_210.pth'
tar = 'work_dirs/den0f_att_16/epoch_210_backbone.pth'
src_dict = torch.load(src, map_location='cpu')
src_dict = src_dict['state_dict']
tar_dict = {}
for key in src_dict:
    if 'backbone.' in key:
        tar_key = key.replace('backbone.', '')
        tar_dict[tar_key] = src_dict[key]
torch.save(tar_dict, tar)