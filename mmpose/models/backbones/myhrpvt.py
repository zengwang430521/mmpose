import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
from .pvt_v2 import (Block, DropPath, DWConv, OverlapPatchEmbed, trunc_normal_, Attention, Mlp)

import pdb
import torch
import torch.nn as nn
from mmcv.cnn import (
    build_conv_layer,
    build_norm_layer,
    constant_init,
    kaiming_init,
    normal_init,
)
from mmcv.runner import load_checkpoint
from mmcv.runner.checkpoint import load_state_dict
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmpose.models.utils.ops import resize
from mmpose.utils import get_root_logger
from ..builder import BACKBONES
from .modules.bottleneck_block import Bottleneck
from .modules.transformer_block import MlpDWBN


from .pvt_v2_3h2 import MyAttention


class MyBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            sr_ratio=1,
            linear=False,
            conv_cfg=None,
            norm_cfg=dict(type="BN", requires_grad=True)
    ):
        super().__init__()
        self.dim = inplanes
        self.out_dim = planes
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.norm1 = norm_layer(self.dim)
        self.attn = MyAttention(
            self.dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(self.dim)
        mlp_hidden_dim = int(self.dim * mlp_ratio)
        self.mlp = MyMlp(
            in_features=self.dim,
            out_features=self.out_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            linear=linear
        )

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

    def forward(self, input_dict):
        x = input_dict['x']
        loc_orig = input_dict['loc_orig']
        x_source = input_dict['x_source']
        idx_agg = input_dict['idx_agg']
        agg_weight = input_dict['agg_weight']
        idx_agg_source = input_dict['idx_agg_source']
        agg_weight_source = input_dict['agg_weight_source']
        H, W = input_dict['map_size']
        conf_source = input_dict['conf_source']

        x1 = x + self.drop_path(self.attn(self.norm1(x),
                                          loc_orig,
                                          self.norm1(x_source),
                                          None,
                                          idx_agg_source,
                                          H, W, conf_source))

        x2 = x1 + self.drop_path(self.mlp(self.norm2(x1),
                                          None,
                                          loc_orig,
                                          idx_agg,
                                          agg_weight,
                                          H, W))
        out_dict = {
            'x': x2,
            'idx_agg': idx_agg,
            'agg_weight': agg_weight,
            'x_source': x2,
            'idx_agg_source': idx_agg,
            'agg_weight_source': agg_weight,
            'loc_orig': loc_orig,
            'map_size': (H, W),
            'conf_source': None,
        }
        return out_dict


def DownUp_Layer(target_dict, source_dict):
    x_s = source_dict['x']
    x_t = target_dict['x']
    idx_agg_s = source_dict['idx_agg']
    idx_agg_t = target_dict['idx_agg']
    agg_weight_t = target_dict['idx_weight']
    B, T, C = x_t.shape
    B, S, C = x_s.shape
    N0 = idx_agg_s.shape[1]

    idx_batch = torch.range(B)[:, None].expand(B, N0)
    coor = torch.stack([idx_batch + idx_agg_t, idx_batch + idx_agg_s], dim=0).reshape(2, B*N0)
    weight = agg_weight_t
    if weight is None:
        weight = x_s.new_ones(B, N0, 1)
    A = torch.sparse.FloatTensor(coor, weight, torch.Size([B*T, B*S]))
    # all_weight = A.type(torch.float32) @ x.new_ones(B*N, 1).type(torch.float32) + 1e-6
    all_weight = A @ x_s.new_ones(B*S, 1) + 1e-6
    weight = weight / all_weight[(idx_batch + idx_agg_t).reshape(-1), 0]

    A = torch.sparse.FloatTensor(coor, weight, torch.Size([B*T, B*S]))
    x_out = A @ x_s.reshape(B*S, C)
    x_out = x_out.reshape(B, T, C)
    return x_out




