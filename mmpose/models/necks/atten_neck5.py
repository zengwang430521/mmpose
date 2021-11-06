# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from ..builder import NECKS
from ..backbones.utils_mine import token2map_agg_mat, downup
import numpy as np
from ..backbones.pvt_v2 import trunc_normal_, DropPath
from ..backbones.pvt_v2_3h2_density import MyMlp, token2map_agg_mat, MyBlock
import math

'''
with sr layers
'''


@NECKS.register_module()
class AttenNeck5(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels=128,
                 num_outs=1,
                 start_level=0,
                 end_level=-1,
                 num_heads=[2, 2, 2, 2],
                 mlp_ratios=[4, 4, 4, 4],
                 sr_ratios=[8, 4, 2, 1],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,

                 add_extra_convs=False,
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
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg

        self.start_level = start_level
        if end_level == -1:
            end_level = len(in_channels) - 1
        self.end_level = end_level

        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.merge_blocks = nn.ModuleList()

        for i in range(self.start_level, self.end_level + 1):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)

        for i in range(self.start_level, self.end_level):
            merge_block = MyBlock(
                dim=out_channels, num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
                sr_ratio=sr_ratios[i]
            )
            self.merge_blocks.append(merge_block)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build lateral tokens
        input_dicts = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            tmp = inputs[i + self.start_level]
            input_dicts.append(
                {'x': lateral_conv(tmp[0].unsqueeze(2).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(2),
                 'map_size': tmp[2],
                 'loc_orig': tmp[3],
                 'idx_agg': tmp[4],
                 'agg_weight': tmp[5]
                 })

        # build gather dict
        tmp = inputs[-1]
        gather_dict = {'x': tmp[0],
                       'map_size': tmp[2],
                       'loc_orig': tmp[3],
                       'idx_agg': tmp[4],
                       'agg_weight': tmp[5]}

        # merge from high levle to low level
        for i in range(len(input_dicts) - 2, -1, -1):
            input_dicts[i]['x'] = input_dicts[i]['x'] + downup(input_dicts[i], input_dicts[i+1])

            x = input_dicts[i]['x']
            idx_agg = input_dicts[i]['idx_agg']
            agg_weight = input_dicts[i]['agg_weight']
            loc_orig = input_dicts[i]['loc_orig']
            H, W = input_dicts[i]['map_size']
            input_dicts[i]['x'] = self.merge_blocks[i](x, idx_agg, agg_weight, loc_orig, x, idx_agg, agg_weight, H, W, conf_source=None)

        out, _ = token2map_agg_mat(
            input_dicts[0]['x'],
            None,
            input_dicts[0]['loc_orig'],
            input_dicts[0]['idx_agg'],
            input_dicts[0]['map_size'],
        )
        return out


'''with norm'''
@NECKS.register_module()
class AttenNeck5N(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels=128,
                 num_outs=1,
                 start_level=0,
                 end_level=-1,
                 num_heads=[2, 2, 2, 2],
                 mlp_ratios=[4, 4, 4, 4],
                 sr_ratios=[8, 4, 2, 1],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,

                 add_extra_convs=False,
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
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg

        self.start_level = start_level
        if end_level == -1:
            end_level = len(in_channels) - 1
        self.end_level = end_level

        self.add_extra_convs = add_extra_convs

        self.lateral_convs = nn.ModuleList()
        self.merge_blocks = nn.ModuleList()

        for i in range(self.start_level, self.end_level + 1):
            # l_conv = ConvModule(
            #     in_channels[i],
            #     out_channels,
            #     1,
            #     conv_cfg=conv_cfg,
            #     norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
            #     act_cfg=act_cfg,
            #     inplace=False)
            # self.lateral_convs.append(l_conv)

            l_conv = nn.Sequential(
                nn.Linear(in_channels[i], out_channels),
                nn.LayerNorm(out_channels)
            )
            self.lateral_convs.append(l_conv)

        for i in range(self.start_level, self.end_level):
            merge_block = MyBlock(
                dim=out_channels, num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
                sr_ratio=sr_ratios[i]
            )
            self.merge_blocks.append(merge_block)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build lateral tokens
        input_dicts = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            tmp = inputs[i + self.start_level]
            input_dicts.append(
                {'x': lateral_conv(tmp[0]),
                 # 'x': lateral_conv(tmp[0].unsqueeze(2).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(2),
                 'map_size': tmp[2],
                 'loc_orig': tmp[3],
                 'idx_agg': tmp[4],
                 'agg_weight': tmp[5]
                 })

        # build gather dict
        tmp = inputs[-1]
        gather_dict = {'x': tmp[0],
                       'map_size': tmp[2],
                       'loc_orig': tmp[3],
                       'idx_agg': tmp[4],
                       'agg_weight': tmp[5]}

        # merge from high levle to low level
        for i in range(len(input_dicts) - 2, -1, -1):
            input_dicts[i]['x'] = input_dicts[i]['x'] + downup(input_dicts[i], input_dicts[i+1])

            x = input_dicts[i]['x']
            idx_agg = input_dicts[i]['idx_agg']
            agg_weight = input_dicts[i]['agg_weight']
            loc_orig = input_dicts[i]['loc_orig']
            H, W = input_dicts[i]['map_size']
            input_dicts[i]['x'] = self.merge_blocks[i](x, idx_agg, agg_weight, loc_orig, x, idx_agg, agg_weight, H, W, conf_source=None)

        out, _ = token2map_agg_mat(
            input_dicts[0]['x'],
            None,
            input_dicts[0]['loc_orig'],
            input_dicts[0]['idx_agg'],
            input_dicts[0]['map_size'],
        )
        return out
