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


class AttentionBN(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False,
                 norm_cfg=dict(type="BN", requires_grad=True)):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.norm_cfg = norm_cfg

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                # self.norm = nn.LayerNorm(dim)
                self.norm = build_norm_layer(norm_cfg, self.dim)[1]

        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            # self.norm = nn.LayerNorm(dim)
            self.norm = build_norm_layer(norm_cfg, self.dim)[1]
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

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.norm(self.sr(x_)).reshape(B, C, -1).permute(0, 2, 1)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.norm(self.sr(self.pool(x_))).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PVT2Block(nn.Module):
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
            norm_cfg=dict(type="BN", requires_grad=True),
    ):
        super().__init__()
        self.dim = inplanes
        self.out_dim = planes
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.norm1 = norm_layer(self.dim)
        self.attn = Attention(
            self.dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(self.dim)
        mlp_hidden_dim = int(self.dim * mlp_ratio)
        self.mlp = Mlp(
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

    def forward(self, x):
        B, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x


class PVT2Block_DWBN(nn.Module):
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
            norm_cfg=dict(type="BN", requires_grad=True),
    ):
        super().__init__()
        self.dim = inplanes
        self.out_dim = planes
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.norm1 = norm_layer(self.dim)
        self.attn = Attention(
            self.dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(self.dim)
        mlp_hidden_dim = int(self.dim * mlp_ratio)
        # self.mlp = Mlp(
        #     in_features=self.dim,
        #     out_features=self.out_dim,
        #     hidden_features=mlp_hidden_dim,
        #     act_layer=act_layer,
        #     drop=drop,
        #     linear=linear
        # )

        self.mlp = MlpDWBN(
            in_features=self.dim,
            hidden_features=mlp_hidden_dim,
            out_features=self.out_dim,
            act_layer=act_layer,
            dw_act_layer=act_layer,
            drop=drop,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
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

    def forward(self, x):
        B, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x


class PVT2Block_BN(nn.Module):
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
            norm_cfg=dict(type="BN", requires_grad=True),
    ):
        super().__init__()
        self.dim = inplanes
        self.out_dim = planes
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # self.norm1 = norm_layer(self.dim)
        self.norm1 = build_norm_layer(norm_cfg, self.dim)[1]

        self.attn = AttentionBN(
            self.dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear,
            norm_cfg=norm_cfg
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = norm_layer(self.dim)
        self.norm2 = build_norm_layer(norm_cfg, self.dim)[1]

        mlp_hidden_dim = int(self.dim * mlp_ratio)
        # self.mlp = Mlp(
        #     in_features=self.dim,
        #     out_features=self.out_dim,
        #     hidden_features=mlp_hidden_dim,
        #     act_layer=act_layer,
        #     drop=drop,
        #     linear=linear
        # )

        self.mlp = MlpDWBN(
            in_features=self.dim,
            hidden_features=mlp_hidden_dim,
            out_features=self.out_dim,
            act_layer=act_layer,
            dw_act_layer=act_layer,
            drop=drop,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
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

    def forward(self, x):
        B, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = x + self.drop_path(self.attn(
            self.norm1(x.permute(0, 2, 1).unsqueeze(-1)).flatten(2).permute(0, 2, 1), H, W))
        x = x + self.drop_path(self.mlp(
            self.norm2(x.permute(0, 2, 1).unsqueeze(-1)).flatten(2).permute(0, 2, 1), H, W))
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        return x


class HRPVTModule(nn.Module):
    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        in_channels,
        num_channels,
        multiscale_output,
        with_cp=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        num_heads=None,
        sr_ratios=None,
        num_mlp_ratios=None,
        drop_paths=0.0,
    ):
        super().__init__()
        self._check_branches(num_branches, num_blocks, in_channels, num_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
            num_heads,
            sr_ratios,
            num_mlp_ratios,
            drop_paths,
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

        # MHSA parameters
        self.num_heads = num_heads
        self.num_mlp_ratios = num_mlp_ratios

    def _check_branches(self, num_branches, num_blocks, in_channels, num_channels):
        logger = get_root_logger()
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(in_channels):
            error_msg = "NUM_BRANCHES({}) <> IN_CHANNELS({})".format(
                num_branches, len(in_channels)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(
        self,
        branch_index,
        block,
        num_blocks,
        num_channels,
        num_heads,
        sr_ratios,
        num_mlp_ratios,
        drop_paths,
        stride=1,
    ):
        """Make one branch."""
        downsample = None
        if (
            stride != 1
            or self.in_channels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    self.in_channels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                build_norm_layer(
                    self.norm_cfg, num_channels[branch_index] * block.expansion
                )[1],
            )

        layers = []

        layers.append(
            block(
                self.in_channels[branch_index],
                num_channels[branch_index],
                num_heads=num_heads[branch_index],
                sr_ratio=sr_ratios[branch_index],
                mlp_ratio=num_mlp_ratios[branch_index],
                drop_path=drop_paths[0],
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
            )
        )
        self.in_channels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.in_channels[branch_index],
                    num_channels[branch_index],
                    num_heads=num_heads[branch_index],
                    sr_ratio=sr_ratios[branch_index],
                    mlp_ratio=num_mlp_ratios[branch_index],
                    drop_path=drop_paths[i],
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(
        self,
        num_branches,
        block,
        num_blocks,
        num_channels,
        num_heads,
        num_window_sizes,
        num_mlp_ratios,
        drop_paths,
    ):
        """Make branches."""
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(
                    i,
                    block,
                    num_blocks,
                    num_channels,
                    num_heads,
                    num_window_sizes,
                    num_mlp_ratios,
                    drop_paths,
                )
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Build fuse layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],
                            nn.Upsample(
                                scale_factor=2 ** (j - i),
                                mode="bilinear",
                                align_corners=False,
                            ),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False,
                                    ),
                                    build_norm_layer(self.norm_cfg, in_channels[j])[1],
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=1,
                                        stride=1,
                                        bias=False,
                                    ),
                                    build_norm_layer(self.norm_cfg, in_channels[i])[1],
                                )
                            )
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False,
                                    ),
                                    build_norm_layer(self.norm_cfg, in_channels[j])[1],
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=1,
                                        stride=1,
                                        bias=False,
                                    ),
                                    build_norm_layer(self.norm_cfg, in_channels[j])[1],
                                    nn.ReLU(inplace=True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y += x[j]
                elif j > i:
                    y = y + resize(
                        self.fuse_layers[i][j](x[j]),
                        size=x[i].shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


@BACKBONES.register_module()
class HRPVT(nn.Module):
    """HRT backbone.
    High Resolution Transformer Backbone
    """

    blocks_dict = {
        "BOTTLENECK": Bottleneck,
        "PVT2BLOCK": PVT2Block,
        "PVT2BLOCK_DWBN": PVT2Block_DWBN,
        "PVT2BLOCK_BN": PVT2Block_BN,

    }

    def __init__(
        self,
        extra,
        in_channels=3,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=False,
        with_cp=False,
        zero_init_residual=False,
    ):
        super().__init__()
        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        # stem net
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, 64, postfix=2)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.add_module(self.norm1_name, norm1)

        self.conv2 = build_conv_layer(
            self.conv_cfg, 64, 64, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)

        # generat drop path rate list
        depth_s2 = (
            self.extra["stage2"]["num_blocks"][0] * self.extra["stage2"]["num_modules"]
        )
        depth_s3 = (
            self.extra["stage3"]["num_blocks"][0] * self.extra["stage3"]["num_modules"]
        )
        depth_s4 = (
            self.extra["stage4"]["num_blocks"][0] * self.extra["stage4"]["num_modules"]
        )
        depths = [depth_s2, depth_s3, depth_s4]
        drop_path_rate = self.extra["drop_path_rate"]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        logger = get_root_logger()
        logger.info(dpr)

        # stage 1
        self.stage1_cfg = self.extra["stage1"]
        num_channels = self.stage1_cfg["num_channels"][0]
        block_type = self.stage1_cfg["block"]
        num_blocks = self.stage1_cfg["num_blocks"][0]

        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)

        # stage 2
        self.stage2_cfg = self.extra["stage2"]
        num_channels = self.stage2_cfg["num_channels"]
        block_type = self.stage2_cfg["block"]

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channels], num_channels
        )
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels, drop_paths=dpr[0:depth_s2]
        )

        # stage 3
        self.stage3_cfg = self.extra["stage3"]
        num_channels = self.stage3_cfg["num_channels"]
        block_type = self.stage3_cfg["block"]

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg,
            num_channels,
            drop_paths=dpr[depth_s2 : depth_s2 + depth_s3],
        )

        # stage 4
        self.stage4_cfg = self.extra["stage4"]
        num_channels = self.stage4_cfg["num_channels"]
        block_type = self.stage4_cfg["block"]

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg,
            num_channels,
            multiscale_output=self.stage4_cfg.get("multiscale_output", False),
            drop_paths=dpr[depth_s2 + depth_s3 :],
        )

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg, num_channels_cur_layer[i])[
                                1
                            ],
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else in_channels
                    )
                    conv_downsamples.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg, out_channels)[1],
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_layer(
        self,
        block,
        inplanes,
        planes,
        blocks,
        stride=1,
        num_heads=1,
        sr_ratio=1,
        mlp_ratio=4.0,
    ):
        """Make each layer."""
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1],
            )

        layers = []

        if block == Bottleneck:
            layers.append(
                block(
                    inplanes,
                    planes,
                    stride,
                    downsample=downsample,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                )
            )
        else:
            layers.append(
                block(
                    inplanes,
                    planes,
                    num_heads=num_heads,
                    sr_ratio=sr_ratio,
                    mlp_ratio=mlp_ratio,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                )
            )

        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                )
            )

        return nn.Sequential(*layers)

    def _make_stage(
        self, layer_config, in_channels, multiscale_output=True, drop_paths=0.0
    ):
        """Make each stage."""
        num_modules = layer_config["num_modules"]
        num_branches = layer_config["num_branches"]
        num_blocks = layer_config["num_blocks"]
        num_channels = layer_config["num_channels"]
        block = self.blocks_dict[layer_config["block"]]

        num_heads = layer_config["num_heads"]
        sr_ratios = layer_config["sr_ratios"]

        num_mlp_ratios = layer_config["num_mlp_ratios"]

        hr_modules = []
        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            hr_modules.append(
                HRPVTModule(
                    num_branches,
                    block,
                    num_blocks,
                    in_channels,
                    num_channels,
                    reset_multiscale_output,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    num_heads=num_heads,
                    sr_ratios=sr_ratios,
                    num_mlp_ratios=num_mlp_ratios,
                    drop_paths=drop_paths[num_blocks[0] * i : num_blocks[0] * (i + 1)],
                )
            )

        return nn.Sequential(*hr_modules), in_channels

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
            Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            ckpt = load_checkpoint(self, pretrained, strict=False)
            if "model" in ckpt:
                msg = self.load_state_dict(ckpt["model"], strict=False)
                logger.info(msg)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    """mmseg: kaiming_init(m)"""
                    normal_init(m, std=0.001)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg["num_branches"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["num_branches"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["num_branches"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        return y_list

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
