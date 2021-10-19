# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS
from ..backbones.utils_mine import token2map_agg_sparse
import numpy as np

from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init, ConvModule)
import torch
from mmpose.models.utils.ops import resize



@NECKS.register_module()
class ResizeCat(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernels=(1, 3, 3, 3),
                 groups=(1, 128, 128, 128),
                 in_index=[0, 1, 2, 3],
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                 inter_mode='bilinear',
    ):
        super().__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.inter_mode = inter_mode
        self.in_index = in_index

        self.in_channels = sum(in_channels)
        layers = []
        for i in range(len(kernels)):
            layers.append(
                ConvModule(
                    self.in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=kernels[i],
                    padding=kernels[i] // 2,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    groups=groups[i])
            )
        self.conv = nn.Sequential(*layers)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        inputs = [inputs[i] for i in self.in_index]
        if isinstance(inputs[0], list) or isinstance(inputs[0], tuple):
            H, W = inputs[0][2]

            upsampled_inputs = [
                token2map_agg_sparse(tmp[0], tmp[1], tmp[3], tmp[4], [H, W])[0]
                for tmp in inputs]
            inputs = torch.cat(upsampled_inputs, dim=1)

        else:
            if self.inter_mode == 'nearest':
                upsampled_inputs = [
                    resize(
                        input=x,
                        size=inputs[0].shape[2:],
                        mode='nearest') for x in inputs
                ]
            else:
                upsampled_inputs = [
                    resize(
                        input=x,
                        size=inputs[0].shape[2:],
                        mode='bilinear',
                        align_corners=self.align_corners) for x in inputs
                ]
            inputs = torch.cat(upsampled_inputs, dim=1)

        inputs = self.conv(inputs)

        return inputs


@NECKS.register_module()
class ResizeCat2(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernels,
                 groups,
                 in_index,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                 inter_mode='bilinear',
                 ):
        super().__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.num_ins = len(in_channels)
        self.inter_mode = inter_mode
        self.in_index = in_index
        self.out_channels = out_channels

        self.in_channels = sum(in_channels)
        conv_layers = []
        for i in range(len(kernels)):
            conv_layers.append(
                ConvModule(
                    in_channels[i],
                    in_channels[i],
                    kernel_size=kernels[i],
                    padding=kernels[i] // 2,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    groups=groups[i])
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.final_layers = ConvModule(
                                sum(in_channels),
                                out_channels,
                                kernel_size=1,
                                padding=0,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg,
                                groups=1)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        inputs = [inputs[i] for i in self.in_index]
        if isinstance(inputs[0], list) or isinstance(inputs[0], tuple):
            H, W = inputs[0][2]
        else:
            H, W = inputs[0].shape[2:]

        outputs = []
        for i in range(len(inputs)):
            if isinstance(inputs[0], list) or isinstance(inputs[0], tuple):
                tmp = inputs[i]
                x = token2map_agg_sparse(tmp[0], tmp[1], tmp[3], tmp[4], [H, W])[0]
            else:
                x = resize(input=inputs[i], size=inputs[0].shape[2:],
                           mode='bilinear', align_corners=self.align_corners)
            x = self.conv_layers[i](x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = self.final_layers(outputs)
        return outputs
