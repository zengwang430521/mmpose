import math
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from .transformer_utils import trunc_normal_, DropPath
from .tc_layers import TCMlp
from .tcformer_utils import token2map, map2token, token_downup
from .tcformer_utils import token2map_flops, map2token_flops, downup_flops
from mmpose.models.builder import NECKS
from mmpose.utils import get_root_logger
import warnings
import torch.nn.functional as F
import copy


class AggregateAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

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

    def forward(self, tar_dict, gather_dict, src_dict=None):
        x = tar_dict['x']
        if src_dict is None:
            src_dict = tar_dict
        x_source = token_downup(gather_dict, src_dict)

        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        x_source = self.norm(x_source)
        x_source = self.act(x_source)
        kv = self.kv(x_source).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AggregateBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AggregateAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TCMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
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

    def forward(self, tar_dict, gather_dict, src_dict=None):
        x = tar_dict['x']
        tar_dict['x'] = self.norm1(x)
        if src_dict is not None:
            src_dict['x'] = self.norm1(src_dict['x'])

        x = x + self.drop_path(self.attn(tar_dict, gather_dict, src_dict))
        H, W = tar_dict['map_size']
        x = x + self.drop_path(self.mlp(self.norm2(x),
                                        tar_dict['loc_orig'],
                                        tar_dict['idx_agg'],
                                        tar_dict['agg_weight'],
                                        H, W))
        return x

# MTA head
@NECKS.register_module()
class MTA(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels=128,
                 num_outs=1,
                 start_level=0,
                 end_level=-1,
                 num_heads=[2, 2, 2, 2],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),

                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 ):
        super().__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.mlp_ratios = mlp_ratios

        self.start_level = start_level
        if end_level == -1:
            end_level = len(in_channels) - 1
        self.end_level = end_level

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
            merge_block = AggregateBlock(
                dim=out_channels, num_heads=num_heads[i], mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=drop_path_rate, norm_layer=norm_layer,
            )
            self.merge_blocks.append(merge_block)



        # add extra conv layers (e.g., RetinaNet)
        self.relu_before_extra_convs = relu_before_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
            self.add_extra_convs = add_extra_convs
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'
        else:
            self.add_extra_convs = 'on_output'

        self.extra_convs = nn.ModuleList()
        extra_levels = num_outs - (self.end_level + 1 - self.start_level)
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.end_level]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.extra_convs.append(extra_fpn_conv)

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

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build lateral tokens
        input_dicts = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            tmp = copy.copy(inputs[i + self.start_level])
            tmp['x'] = lateral_conv(tmp['x'].unsqueeze(2).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).squeeze(2)
            input_dicts.append(tmp)

        # build gather dict
        gather_dict = inputs[-1]

        # merge from high level to low level
        for i in range(len(input_dicts) - 2, -1, -1):
            input_dicts[i]['x'] = input_dicts[i]['x'] + token_downup(input_dicts[i], input_dicts[i+1])
            input_dicts[i]['x'] = self.merge_blocks[i](input_dicts[i], gather_dict, input_dicts[i])

        # out, _ = token2map(
        #     input_dicts[0]['x'],
        #     None,
        #     input_dicts[0]['loc_orig'],
        #     input_dicts[0]['idx_agg'],
        #     input_dicts[0]['map_size'],
        # )
        # return multiple stage feature map

        outs = [
            token2map(
                input_dicts[i]['x'],
                None,
                input_dicts[i]['loc_orig'],
                input_dicts[i]['idx_agg'],
                input_dicts[i]['map_size'],
            )[0] for i in range(len(input_dicts))
        ]

        # part 2: add extra levels
        used_backbone_levels = len(outs)
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))

            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    tmp = inputs[self.end_level]
                    extra_source, _ = token2map(tmp['x'], None, tmp['loc_orig'],
                                             tmp['idx_agg'], tmp['map_size'], tmp['agg_weight'])
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError

                outs.append(self.extra_convs[0](extra_source))
                for i in range(1, self.num_outs - used_backbone_levels):
                    if self.relu_before_extra_convs:
                        outs.append(self.extra_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.extra_convs[i](outs[-1]))
        return outs

    def get_extra_flops(self, h, w):
        flops = 0
        C = self.out_channels
        N0 = h * w
        N4 = N0 // 4 // 4 // 4
        for i in range(3):
            # down up
            flops += downup_flops(N0, C)
            # token and map
            flops += token2map_flops(N0, C) + token2map_flops(N0, C * self.mlp_ratios[i]) + map2token_flops(N0, C * self.mlp_ratios[i])
            # attn
            flops += 2 * N0 * N4 * C
        return flops
