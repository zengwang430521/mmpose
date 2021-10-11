# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbones.utils_mine import token2map, token2map_agg_sparse
from ..builder import NECKS


@NECKS.register_module()
class UpSampleNeck(nn.Module):
    """Upsample
    """

    def __init__(self, scale_factor=8, mode='bilinear', stage=-1):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.stage = stage

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, list):
            x = inputs[self.stage]
        else:
            x = inputs
        outs = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return outs


@NECKS.register_module()
class TokenInterNeck1(nn.Module):
    """Token inter to feature map
    """
    def __init__(self, scale_factor, stage=-1):
        super().__init__()
        self.scale_factor = scale_factor
        self.stage = stage

    def init_weights(self):
        pass

    def forward(self, inputs):
        input = inputs[self.stage]
        x, loc, map_size = input
        h, w = map_size
        h, w = int(h * self.scale_factor), int(w * self.scale_factor)        
        outs = token2map(x, loc, [h, w], kernel_size=self.scale_factor + 1, sigma=2)
        return outs


@NECKS.register_module()
class TokenInterNeck2(nn.Module):
    """Token inter to feature map, merge
    """
    def __init__(self, scale_factor, stage=-1, kernel=1, sigma=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.stage = stage
        self.kernel = kernel
        self.sigma = sigma

    def init_weights(self):
        pass

    def forward(self, inputs):
        input = inputs[self.stage]
        x, loc, map_size, loc_orig, idx_agg = input
        h, w = map_size
        scale_orig = (idx_agg.shape[1] // x.shape[1]) ** 0.5

        if self.scale_factor <= scale_orig:
            h, w = int(h * self.scale_factor), int(w * self.scale_factor)
            outs, _ = token2map_agg_sparse(x, loc, loc_orig, idx_agg, [h, w], kernel=self.kernel, sigma=self.sigma)
        else:
            h, w = int(h * scale_orig), int(w * scale_orig)
            outs, _ = token2map_agg_sparse(x, loc, loc_orig, idx_agg, [h, w], kernel=self.kernel, sigma=self.sigma)
            h, w = map_size
            h, w = int(h * self.scale_factor), int(w * self.scale_factor)
            outs = F.interpolate(outs, [h, w], mode='bilinear')
        return outs

