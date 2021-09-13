import torch

src = 'data/pretrained/my20_2_300.pth'
dst = 'data/pretrained/my20_2_300.pth'
data_dict = torch.load(src)
if 'state_dict' not in data_dict.keys():
    data_dict['state_dict'] = data_dict['model']
torch.save(data_dict, dst)