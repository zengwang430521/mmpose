import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_upsample_layer, constant_init,
                      normal_init)
from torch.nn import functional as F

from ..registry import HEADS
from .top_down_base_head import TopDownBaseHead


@HEADS.register_module()
class RootNetHead(TopDownBaseHead):
    """RootNet.

    paper ref: Gyeongsik Moon, et al. ``Camera Distance-aware Top-down Approach
    for 3D Multi-person  Pose Estimation from a Single RGB Image.''.
    """

    def __init__(self,
                 in_channels=2048,
                 out_channels=256,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4)):
        super().__init__()

        self.in_channels = in_channels
        self.out_channles = out_channels

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')

        self.xy_layer = build_conv_layer(
            cfg=dict(type='Conv2d'),
            in_channels=self.out_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0)

        self.depth_layer = build_conv_layer(
            cfg=dict(type='Conv2d'),
            in_channels=self.in_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    def forward(self, x, k_value):
        output_shape = self.output_shape
        # x, y
        xy = self.deconv_layers(x)
        xy = self.xy_layer(xy)
        xy = xy.view(-1, 1, output_shape[0] * output_shape[1])
        xy = F.softmax(xy, 2)
        xy = xy.view(-1, 1, output_shape[0], output_shape[1])

        hm_x = xy.sum(dim=(2))
        hm_y = xy.sum(dim=(3))

        coord_x = hm_x * torch.arange(output_shape[1]).float().cuda()
        coord_y = hm_y * torch.arange(output_shape[0]).float().cuda()

        coord_x = coord_x.sum(dim=2)
        coord_y = coord_y.sum(dim=2)

        # z
        # global average pooling
        img_feat = torch.mean(
            x.view(x.size(0), x.size(1),
                   x.size(2) * x.size(3)), dim=2)
        img_feat = torch.unsqueeze(img_feat, 2)
        img_feat = torch.unsqueeze(img_feat, 3)
        gamma = self.depth_layer(img_feat)
        gamma = gamma.view(-1, 1)
        depth = gamma * k_value.view(-1, 1)

        coord = torch.cat((coord_x, coord_y, depth), dim=1)
        return coord

    def init_weights(self):
        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.xy_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.depth_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
