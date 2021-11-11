# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmcv import Config

from mmpose.models import build_posenet

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')






if len(args.shape) == 1:
    input_shape = (3, args.shape[0], args.shape[0])
elif len(args.shape) == 2:
    input_shape = (3, ) + tuple(args.shape)
else:
    raise ValueError('invalid input shape')

cfg = Config.fromfile(args.config)
model = build_posenet(cfg.model)
model = model.cuda()
model.eval()

if hasattr(model, 'forward_dummy'):
    model.forward = model.forward_dummy
else:
    raise NotImplementedError(
        'FLOPs counter is currently not currently supported with {}'.
        format(model.__class__.__name__))

flops, params = get_model_complexity_info(model, input_shape)
split_line = '=' * 30
print(f'{split_line}\nInput shape: {input_shape}\n'
      f'Flops: {flops}\nParams: {params}\n{split_line}')
print('!!!Please be cautious if you use the results in papers. '
      'You may need to check if all ops are supported and verify that the '
      'flops computation is correct.')


