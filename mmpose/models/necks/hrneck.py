# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS
# from ..backbones.utils_mine import token2map_agg_mat, downup
from ..backbones.utils_mine import token2map_agg_sparse as token2map

import numpy as np
from .atten_neck3 import _init_weights


@NECKS.register_module()
class HRNeck(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                 ):
        super().__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.num_ins = len(in_channels)
        # self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(len(in_channels)):
            l_conv = nn.Sequential(
                nn.Linear(in_channels[i], out_channels),
                nn.LayerNorm(out_channels)
            )

            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        tokens = [tmp[0] for tmp in inputs]
        for i in range(4):
            print(tokens[i].shape)

        # build laterals in token format
        lateral_tokens = [
            lateral_conv(tokens[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path from token to feature map
        laterals = [None for _ in lateral_tokens]
        used_backbone_levels = len(laterals)

        for i in range(used_backbone_levels):
            tmp = inputs[i]
            map_size = tmp[2]
            for j in range(used_backbone_levels):
                tmp = inputs[j]
                loc, loc_orig, idx_agg = tmp[1], tmp[3], tmp[4]
                if j == 0:
                    laterals[i] = token2map(lateral_tokens[j], loc, loc_orig, idx_agg, map_size, weight=None)[0]
                else:
                    laterals[i] += token2map(lateral_tokens[j], loc, loc_orig, idx_agg, map_size, weight=None)[0]


        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        return outs


