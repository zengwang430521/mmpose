# Copyright (c) OpenMMLab. All rights reserved.

import copy

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.runner import BaseModule
from timm.models.layers import to_2tuple, trunc_normal_
from torch.nn import functional as F

from ...builder import BACKBONES
from ..hrnet import BasicBlock, Bottleneck, HRModule, HRNet

from .tc_layers import TCWinBlock
from .tcformer_utils import (
    map2token, token2map, token_downup, get_grid_loc,
    token_cluster_part_pad, token_cluster_part_follow,
    show_tokens_merge, token_cluster_grid, pca_feature,
    token_remerge_part
)
import math

vis = False
vis = True

# part wise merge with padding with dict as input and output
# no block in this layer, use BN layer.
class CTM_partpad_dict(nn.Module):
    def __init__(self, sample_ratio, embed_dim, dim_out, drop_rate,
                 k=5, nh=1, nw=None, nh_list=None, nw_list=None,
                 use_agg_weight=True, agg_weight_detach=False, with_act=True,
                 ):
        super().__init__()
        # self.sample_num = sample_num
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out

        self.conv = nn.Conv2d(embed_dim, dim_out, kernel_size=3, stride=2, padding=1)
        self.conv_skip = nn.Linear(embed_dim, dim_out, bias=False)
        self.norm = nn.LayerNorm(self.dim_out)
        self.conf = nn.Linear(self.dim_out, 1)

        # for density clustering
        self.k = k

        # for partwise
        self.nh = nh
        self.nw = nw or nh
        self.nh_list = nh_list
        self.nw_list = nw_list or nh_list
        self.use_agg_weight = use_agg_weight
        self.agg_weight_detach = agg_weight_detach
        self.with_act = with_act
        if self.with_act:
            self.act = nn.ReLU(inplace=False)

    def forward(self, input_dict):
        input_dict = input_dict.copy()
        x = input_dict['x']
        loc_orig = input_dict['loc_orig']
        idx_agg = input_dict['idx_agg']
        agg_weight = input_dict['agg_weight']
        H, W = input_dict['map_size']

        if not self.use_agg_weight:
            agg_weight = None

        if agg_weight is not None and self.agg_weight_detach:
            agg_weight = agg_weight.detach()

        B, N, C = x.shape
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = self.norm(x)
        conf = self.conf(x)
        weight = conf.exp()
        input_dict['x'] = x

        B, N, C = x.shape
        sample_num = max(math.ceil(N * self.sample_ratio), 1)
        nh, nw = self.nh, self.nw
        num_part = nh * nw
        sample_num = round(sample_num // num_part) * num_part

        # print('ONLY FOR DEBUG')
        # Ns = x_map.shape[1] * x_map.shape[2]
        # x_down, idx_agg_down, weight_t, _ = token_cluster_grid(input_dict, Ns, conf, weight=None, k=5)

        if self.nh_list is not None and self.nw_list is not None:
            x_down, idx_agg_down, weight_t = token_cluster_part_pad(
                input_dict, sample_num, weight=weight, k=self.k,
                nh_list=self.nh_list, nw_list=self.nw_list
            )
        else:
            x_down, idx_agg_down, weight_t = token_cluster_part_follow(
                input_dict, sample_num, weight=weight, k=self.k, nh=nh, nw=nw
            )

        if agg_weight is not None:
            agg_weight_down = agg_weight * weight_t
            agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]
            if self.agg_weight_detach:
                agg_weight_down = agg_weight_down.detach()
        else:
            agg_weight_down = None

        _, _, H, W = x_map.shape
        input_dict['conf'] = conf
        input_dict['map_size'] = [H, W]

        out_dict = {
            'x': x_down,
            'idx_agg': idx_agg_down,
            'agg_weight': agg_weight_down,
            'loc_orig': loc_orig,
            'map_size': [H, W]
        }

        if self.with_act:
            out_dict['x'] = self.act(out_dict['x'])
            input_dict['x'] = self.act(input_dict['x'])

        return out_dict, input_dict


# part wise merge with padding with dict as input and output
# no block in this layer, use BN layer.
# class CTM_partpad_dict_BN_old(nn.Module):
#     def __init__(self, sample_ratio, embed_dim, dim_out, drop_rate,
#                  k=5, nh=1, nw=None, nh_list=None, nw_list=None,
#                  use_agg_weight=True, agg_weight_detach=False, with_act=True,
#                  norm_cfg=None, remain_res=False
#                  ):
#         super().__init__()
#         # self.sample_num = sample_num
#         self.sample_ratio = sample_ratio
#         self.dim_out = dim_out
#         self.norm_cfg = norm_cfg
#
#         self.conv = nn.Conv2d(embed_dim, dim_out, kernel_size=3, stride=2, padding=1)
#         self.conv_skip = nn.Linear(embed_dim, dim_out, bias=False)
#         self.norm_name, self.norm = build_norm_layer(self.norm_cfg, self.dim_out)
#         self.conf = nn.Linear(self.dim_out, 1)
#
#         # for density clustering
#         self.k = k
#
#         # for partwise
#         self.nh = nh
#         self.nw = nw or nh
#         self.nh_list = nh_list
#         self.nw_list = nw_list or nh_list
#         self.use_agg_weight = use_agg_weight
#         self.agg_weight_detach = agg_weight_detach
#         self.with_act = with_act
#         if self.with_act:
#             self.act = nn.ReLU(inplace=False)
#         self.remain_res = remain_res
#
#     def forward(self, input_dict):
#         input_dict = input_dict.copy()
#         x = input_dict['x']
#         loc_orig = input_dict['loc_orig']
#         idx_agg = input_dict['idx_agg']
#         agg_weight = input_dict['agg_weight']
#         H, W = input_dict['map_size']
#
#         if not self.use_agg_weight:
#             agg_weight = None
#
#         if agg_weight is not None and self.agg_weight_detach:
#             agg_weight = agg_weight.detach()
#
#         B, N, C = x.shape
#         x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])
#
#         x_map = self.conv(x_map)
#         x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + self.conv_skip(x)
#         x = token_norm(self.norm, self.norm_name, x)
#         # if self.with_act:
#         #     x = self.act(x)
#
#         conf = self.conf(x)
#         weight = conf.exp()
#         input_dict['x'] = x
#
#         B, N, C = x.shape
#         sample_num = max(math.ceil(N * self.sample_ratio), 1)
#         nh, nw = self.nh, self.nw
#         num_part = nh * nw
#         sample_num = round(sample_num // num_part) * num_part
#
#         # print('ONLY FOR DEBUG')
#         # Ns = x_map.shape[1] * x_map.shape[2]
#         # x_down, idx_agg_down, weight_t, _ = token_cluster_grid(input_dict, Ns, conf, weight=None, k=5)
#
#
#         # print('for debug only')
#         # import matplotlib.pyplot as plt
#         # plt.subplot(1, 2, 2)
#         # tmp = pca_feature(input_dict['x'].detach().float())
#         # tmp_map = token2map(tmp, None, input_dict['loc_orig'], input_dict['idx_agg'], input_dict['map_size'])[0]
#         # plt.imshow(tmp_map[0].permute(1, 2, 0).detach().cpu().float())
#
#
#         if self.nh_list is not None and self.nw_list is not None:
#             x_down, idx_agg_down, weight_t = token_cluster_part_pad(
#                 input_dict, sample_num, weight=weight, k=self.k,
#                 nh_list=self.nh_list, nw_list=self.nw_list
#             )
#         else:
#             x_down, idx_agg_down, weight_t = token_cluster_part_follow(
#                 input_dict, sample_num, weight=weight, k=self.k, nh=nh, nw=nw
#             )
#
#         if agg_weight is not None:
#             agg_weight_down = agg_weight * weight_t
#             agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]
#             if self.agg_weight_detach:
#                 agg_weight_down = agg_weight_down.detach()
#         else:
#             agg_weight_down = None
#
#         _, _, H, W = x_map.shape
#         input_dict['conf'] = conf
#         if not self.remain_res:
#             input_dict['map_size'] = [H, W]
#
#         out_dict = {
#             'x': x_down,
#             'idx_agg': idx_agg_down,
#             'agg_weight': agg_weight_down,
#             'loc_orig': loc_orig,
#             'map_size': [H, W]
#         }
#
#         # print('ONLY FOR DEBUG.')
#         # xt = x_down.permute(0, 2, 1).reshape(x_map.shape)
#         # xt = token2map(x, None, loc_orig, idx_agg, [H, W])[0]
#
#         if self.with_act:
#             out_dict['x'] = self.act(out_dict['x'])
#             input_dict['x'] = self.act(input_dict['x'])
#
#         return out_dict, input_dict


# part wise merge with padding with dict as input and output
# no block in this layer, use BN layer.
class CTM_partpad_dict_BN(nn.Module):
    def __init__(self, sample_ratio, embed_dim, dim_out, drop_rate,
                 k=5, nh_list=None, nw_list=None, level=0,
                 use_agg_weight=True, agg_weight_detach=False, with_act=True,
                 norm_cfg=None, remain_res=False, cluster=False, first_cluster=False
                 ):
        super().__init__()
        # self.sample_num = sample_num
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out
        self.norm_cfg = norm_cfg

        self.conv = nn.Conv2d(embed_dim, dim_out, kernel_size=3, stride=2, padding=1)
        self.conv_skip = nn.Linear(embed_dim, dim_out, bias=False)
        self.norm_name, self.norm = build_norm_layer(self.norm_cfg, self.dim_out)

        self.conf = nn.Linear(self.dim_out, 1)

        # for density clustering
        self.k = k

        # for partwise
        self.nh_list = nh_list
        self.nw_list = nw_list or nh_list
        self.level = level
        self.use_agg_weight = use_agg_weight
        self.agg_weight_detach = agg_weight_detach
        self.with_act = with_act
        if self.with_act:
            self.act = nn.ReLU(inplace=False)
        self.remain_res = remain_res
        self.cluster = cluster
        self.have_cluster = first_cluster

    def forward(self, input_dict):
        input_dict = input_dict.copy()
        x = input_dict['x']
        loc_orig = input_dict['loc_orig']
        idx_agg = input_dict['idx_agg']
        agg_weight = input_dict['agg_weight']
        H, W = input_dict['map_size']

        if not self.use_agg_weight:
            agg_weight = None

        if agg_weight is not None and self.agg_weight_detach:
            agg_weight = agg_weight.detach()

        B, N, C = x.shape
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token(x_map, N, loc_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = token_norm(self.norm, self.norm_name, x)
        # if self.with_act:
        #     x = self.act(x)

        conf = self.conf(x)
        weight = conf.exp()
        input_dict['x'] = x
        B, N, C = x.shape

        # print('ONLY FOR DEBUG')
        # Ns = x_map.shape[1] * x_map.shape[2]
        # x_down, idx_agg_down, weight_t, _ = token_cluster_grid(input_dict, Ns, conf, weight=None, k=5)


        # print('for debug only')
        # import matplotlib.pyplot as plt
        # plt.subplot(1, 2, 2)
        # tmp = pca_feature(input_dict['x'].detach().float())
        # tmp_map = token2map(tmp, None, input_dict['loc_orig'], input_dict['idx_agg'], input_dict['map_size'])[0]
        # plt.imshow(tmp_map[0].permute(1, 2, 0).detach().cpu().float())

        if self.cluster:
            sample_num = max(math.ceil(N * self.sample_ratio), 1)
            nh, nw = self.nh_list[self.level], self.nw_list[self.level]
            num_part = nh * nw
            sample_num = round(sample_num // num_part) * num_part

            x_down, idx_agg_down, agg_weight_down = token_remerge_part(
                input_dict, Ns=sample_num,  weight=weight, k=self.k,
                nh_list=self.nh_list, nw_list=self.nw_list, level=self.level,
                output_tokens=True, first_cluster=self.have_cluster)
        else:
            x_down, idx_agg_down, weight_t, _ = token_cluster_grid(
                input_dict, Ns=None, conf=None, weight=weight)

            agg_weight_down = agg_weight * weight_t
            agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]

        if self.agg_weight_detach:
            agg_weight_down = agg_weight_down.detach()

        _, _, H, W = x_map.shape
        input_dict['conf'] = conf
        if not self.remain_res:
            input_dict['map_size'] = [H, W]

        out_dict = {
            'x': x_down,
            'idx_agg': idx_agg_down,
            'agg_weight': agg_weight_down,
            'loc_orig': loc_orig,
            'map_size': [H, W]
        }

        # print('ONLY FOR DEBUG.')
        # xt = x_down.permute(0, 2, 1).reshape(x_map.shape)
        # xt = token2map(x, None, loc_orig, idx_agg, [H, W])[0]

        if self.with_act:
            out_dict['x'] = self.act(out_dict['x'])
            input_dict['x'] = self.act(input_dict['x'])

        return out_dict, input_dict


class DictLayer(nn.Module):
    def __init__(self, layer, input_decap=False, output_cap=True):
        super().__init__()
        self.layer = layer
        self.input_decap = input_decap
        self.output_cap = output_cap

    def forward(self, input_dict):
        if self.input_decap:
            x = self.layer(input_dict['x'])
        else:
            x = self.layer(input_dict)

        if self.output_cap:
            out_dict = input_dict.copy()
            out_dict['x'] = x
            return out_dict
        else:
            return x


class TokenConv(nn.Conv2d):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        groups = kwargs['groups'] if 'groups' in kwargs.keys() else 1
        self.skip = nn.Conv1d(in_channels=kwargs['in_channels'],
                              out_channels=kwargs['out_channels'],
                              kernel_size=1, bias=False,
                              groups=groups)

    def forward(self, input_dict):
        x = input_dict['x']
        loc_orig = input_dict['loc_orig']
        idx_agg = input_dict['idx_agg']
        agg_weight = input_dict['agg_weight']
        H, W = input_dict['map_size']

        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])
        x_map = super().forward(x_map)
        x = map2token(x_map, x.shape[1], loc_orig, idx_agg, agg_weight) + self.skip(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


def token_norm(norm_layer, norm_name, x):
    if 'ln' in norm_name:
        x = norm_layer(x)
    else:
        x = norm_layer(x.permute(0, 2, 1).unsqueeze(-1)).flatten(2).permute(0, 2, 1)
    return x


class TokenNorm(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.name = norm[0]
        self.norm = norm[1]

    def forward(self, x):
        if 'ln' in self.name:
            x = self.norm(x)
        else:
            x = self.norm(x.permute(0, 2, 1).unsqueeze(-1)).flatten(2).permute(0, 2, 1)
        return x


class TokenDownLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg,
                 norm_cfg,
                 with_act=True,
                 ):
        super().__init__()
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.dw_conv = build_conv_layer(
            self.conv_cfg,
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=in_channels,
            bias=False)

        self.dw_skip = build_conv_layer(
            self.conv_cfg,
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            groups=in_channels,
            bias=False)

        self.norm1 = build_norm_layer(self.norm_cfg, in_channels)[1]
        self.conv = build_conv_layer(
            self.conv_cfg,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False)
        self.norm2 = build_norm_layer(self.norm_cfg, out_channels)[1]
        self.with_act = with_act
        if self.with_act:
            self.act = nn.ReLU(inplace=True)

    def forward(self, input_dict, tar_dict):
        x = input_dict['x']
        loc_orig = input_dict['loc_orig']
        idx_agg = input_dict['idx_agg']
        agg_weight = input_dict['agg_weight']
        H, W = input_dict['map_size']

        Nt = tar_dict['x'].shape[1]
        idx_agg_t = tar_dict['idx_agg']
        agg_weight_t = tar_dict['agg_weight']

        # real 2D
        x_map, _ = token2map(x, None, loc_orig, idx_agg, [H, W])
        x_map = self.dw_conv(x_map)
        x_map = map2token(x_map, Nt, loc_orig, idx_agg_t, agg_weight_t)

        # fake 2D
        x = token_downup(source_dict=input_dict, target_dict=tar_dict)
        x = x.permute(0, 2, 1)[..., None]
        x = self.dw_skip(x)
        x = x + x_map.permute(0, 2, 1)[..., None]

        x = self.norm1(x)
        x = self.conv(x)
        x = self.norm2(x)
        x = x.squeeze(-1).permute(0, 2, 1)

        if self.with_act:
            x = self.act(x)

        out_dict = tar_dict.copy()
        out_dict['x'] = x
        return out_dict


# one step for multi-level sampling
class TokenFuseLayer(nn.Module):
    def __init__(
            self,
            num_branches,
            in_channels,
            multiscale_output,
            conv_cfg,
            norm_cfg,
            remerge=False,
            ignore_density=False,
            bilinear_upsample=False,
            k=5, nh_list=None, nw_list=None,
    ):
        super().__init__()
        # for remerge
        self.remerge = remerge
        self.ignore_density = ignore_density
        self.nh_list = nh_list
        self.nw_list = nw_list
        self.k = k


        self.bilinear_upsample = bilinear_upsample
        self.norm_cfg = norm_cfg
        self.num_branches = num_branches
        self.num_out_branches = num_branches if multiscale_output else 1
        fuse_layers = []
        for i in range(self.num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    # upsample
                    fuse_link = DictLayer(
                        nn.Sequential(
                            nn.Linear(in_channels[j], in_channels[i], bias=False),
                            TokenNorm(build_norm_layer(self.norm_cfg, in_channels[i]))),
                        input_decap=True,
                        output_cap=True,
                    )
                elif j == i:
                    # same stage
                    fuse_link = None
                else:
                    # down sample
                    fuse_link = []
                    for k in range(i - j):
                        fuse_link.append(
                            TokenDownLayer(
                                in_channels=in_channels[j],
                                out_channels=in_channels[i] if k == i - j - 1 else in_channels[j],
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                with_act=(k != i - j - 1)),
                        )
                    fuse_link = nn.ModuleList(fuse_link)

                fuse_layer.append(fuse_link)
            fuse_layer = nn.ModuleList(fuse_layer)
            fuse_layers.append(fuse_layer)
        self.fuse_layers = nn.ModuleList(fuse_layers)
        self.relu = nn.ReLU(inplace=True)



    def forward(self, input_lists):
        assert len(input_lists) == self.num_branches
        out_lists = []

        # print('only for debug!')
        # # we need re-cluster and re-merge tokens from the first level to the last
        # for i in range(self.num_out_branches):
        #     tar_dict = input_lists[i]
        #
        #     if i > 0 and self.remerge:
        #         # remerge
        #         x, idx_agg, agg_weight = token_remerge_part(
        #             out_lists[-1],
        #             Ns=tar_dict['x'].shape[1],
        #             weight=None,
        #             k=self.k,
        #             nh_list=self.nh_list,
        #             nw_list=self.nw_list,
        #             level=i,
        #             output_tokens=True,
        #             first_cluster=(i == 1),
        #         )
        #         B, _, C = tar_dict['x'].shape
        #         # x = tar_dict['x'].new_zeros([B, x.shape[1], C])
        #
        #     else:
        #         # NOT remerge
        #         x = tar_dict['x']
        #         idx_agg = tar_dict['idx_agg']
        #         agg_weight = tar_dict['agg_weight']
        #
        #     out_dict = {
        #         'x': x,
        #         'idx_agg': idx_agg,
        #         'agg_weight': agg_weight,
        #         'map_size': tar_dict['map_size'],
        #         'loc_orig': tar_dict['loc_orig']
        #     }
        #
        #     # append a fake output for downsample
        #     out_lists.append(out_dict)
        #
        # for i in range(self.num_out_branches):
        #     tar_dict = input_lists[i]
        #     if i > 0 and self.remerge:
        #         out_lists[i]['x'] = tar_dict['x'].new_zeros(tar_dict['x'].shape)



        # we need re-cluster and re-merge tokens from the first level to the last
        for i in range(self.num_out_branches):


            tar_dict = input_lists[i]

            if i > 0 and self.remerge:
                # remerge
                x, idx_agg, agg_weight = token_remerge_part(
                    out_lists[-1],
                    Ns=tar_dict['x'].shape[1],
                    weight=None,
                    k=self.k,
                    nh_list=self.nh_list,
                    nw_list=self.nw_list,
                    level=i,
                    output_tokens=False,
                    first_cluster=(i == 1),
                    ignore_density=self.ignore_density
                )
                B, _, C = tar_dict['x'].shape
                x = tar_dict['x'].new_zeros([B, x.shape[1], C])

            else:
                # NOT remerge
                x = tar_dict['x']
                idx_agg = tar_dict['idx_agg']
                agg_weight = tar_dict['agg_weight']

            out_dict = {
                'x': x,
                'idx_agg': idx_agg,
                'agg_weight': agg_weight,
                'map_size': tar_dict['map_size'],
                'loc_orig': tar_dict['loc_orig']
            }

            # append a fake output for downsample
            out_lists.append(out_dict)

            # source loop
            out_dict = out_lists[i]
            for j in range(self.num_branches):
                src_dict = input_lists[j].copy()

                if j > i:
                    # upsample, just one step
                    src_dict = self.fuse_layers[i][j](src_dict)
                    if self.bilinear_upsample:
                        x_tmp, _ = token2map(
                            src_dict['x'],
                            None,
                            src_dict['loc_orig'],
                            src_dict['idx_agg'],
                            out_dict['map_size'],
                        )
                        avg_k = 2 ** (j - i) + 1
                        pad = (avg_k - 1) // 2
                        x_tmp = F.avg_pool2d(F.pad(x_tmp, [pad, pad, pad, pad], mode='replicate'),
                                             kernel_size=avg_k, stride=1, padding=0)
                        # x2 = F.avg_pool2d(x_tmp, kernel_size=avg_k, stride=1, padding=pad, count_include_pad=False)
                        x_tmp = map2token(
                            x_tmp,
                            out_dict['x'].shape[1],
                            out_dict['loc_orig'],
                            out_dict['idx_agg'],
                            out_dict['agg_weight'])
                    else:
                        x_tmp = token_downup(target_dict=out_dict, source_dict=src_dict)

                    out_dict['x'] = out_dict['x'] + x_tmp

                elif j == i:
                    # the same level
                    if i > 0 and self.remerge:
                        out_dict['x'] = out_dict['x'] + token_downup(out_dict, src_dict)

                else:
                    # down sample
                    fuse_link = self.fuse_layers[i][j]
                    for k in range(i - j):
                        tar_dict = out_lists[k + j + 1]
                        src_dict = fuse_link[k](src_dict, tar_dict)
                    out_dict['x'] = out_dict['x'] + src_dict['x']

            out_dict['x'] = self.relu(out_dict['x'])
            # replace fake dict with real dict
            # out_lists[-1] = out_dict
            out_lists[i] = out_dict

        return out_lists


class HRTCModule(HRModule):

    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 in_channels,
                 num_channels,
                 multiscale_output,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 num_heads=None,
                 num_window_sizes=None,
                 num_mlp_ratios=None,
                 drop_paths=0.0,
                 upsample_cfg=dict(mode='bilinear', align_corners=False),
                 bilinear_upsample=False,
                 attn_type='window',
                 nh_list=[1, 1, 1, 1],
                 nw_list=[1, 1, 1, 1],
                 input_with_cluster=None,
                 remerge=False,
                 ignore_density=False,
                 k=5,
                 ):

        super(HRModule, self).__init__()
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        self._check_branches(num_branches, num_blocks, in_channels,
                             num_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches
        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
        self.upsample_cfg = upsample_cfg
        self.num_heads = num_heads
        self.num_window_sizes = num_window_sizes
        self.num_mlp_ratios = num_mlp_ratios
        self.in_channels = in_channels
        self.num_branches = num_branches
        self.drop_paths = drop_paths
        self.bilinear_upsample = bilinear_upsample
        self.attn_type = attn_type

        if input_with_cluster is None:
            self.after_cluster = [False, True, True, True]
        else:
            self.after_cluster = input_with_cluster
        self.nh_list = nh_list
        self.nw_list = nw_list
        self.remerge = remerge
        self.ignore_density = ignore_density
        self.k = k

        self.branches = self._make_branches(
            num_branches,
            blocks,
            num_blocks,
            num_channels,
            num_heads,
            num_window_sizes,
            num_mlp_ratios,
            drop_paths,
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _make_one_branch(
        self,
        branch_index,
        block,
        num_blocks,
        num_channels,
        num_heads,
        num_window_sizes,
        num_mlp_ratios,
        drop_paths,
        stride=1,
        attn_type='window',
        nh=1,
        nw=1,
        after_cluster=False
    ):
        """Make one branch."""
        # LocalWindowTransformerBlock does not support down sample layer yet.
        assert stride == 1 and self.in_channels[branch_index] == num_channels[
            branch_index]
        layers = []
        for i in range(num_blocks[branch_index]):
            layers.append(
                block(
                    self.in_channels[branch_index],
                    num_channels[branch_index],
                    num_heads=num_heads[branch_index],
                    window_size=num_window_sizes[branch_index],
                    mlp_ratio=num_mlp_ratios[branch_index],
                    drop_path=drop_paths[i],
                    mlp_norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    attn_type=attn_type,
                    num_parts=(nh, nw),
                    after_cluster=after_cluster,
                ))
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
                    attn_type=self.attn_type,
                    nh=self.nh_list[i],
                    nw=self.nw_list[i],
                    after_cluster=self.after_cluster[i]
                ))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Build fuse layer."""
        if self.num_branches == 1:
            return None
        return TokenFuseLayer(
            self.num_branches,
            self.in_channels,
            self.multiscale_output,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            bilinear_upsample=self.bilinear_upsample,
            remerge=self.remerge,
            ignore_density=self.ignore_density,
            nh_list=self.nh_list,
            nw_list=self.nw_list,
            k=self.k
        )

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = self.fuse_layers(x)
        return x_fuse



@BACKBONES.register_module()
class HRTCFormer(HRNet):
    """HRFormer backbone.

    High Resolution Transformer Backbone
    """

    blocks_dict = {
        'BASIC': BasicBlock,
        'BOTTLENECK': Bottleneck,
        'TCWINBLOCK': TCWinBlock
    }

    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=False,
                 return_map=False):
        super(HRNet, self).__init__()

        # generate drop path rate list
        depth_s2 = (
            extra['stage2']['num_blocks'][0] * extra['stage2']['num_modules'])
        depth_s3 = (
            extra['stage3']['num_blocks'][0] * extra['stage3']['num_modules'])
        depth_s4 = (
            extra['stage4']['num_blocks'][0] * extra['stage4']['num_modules'])
        depths = [depth_s2, depth_s3, depth_s4]
        drop_path_rate = extra['drop_path_rate']
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        extra['stage2']['drop_path_rates'] = dpr[0:depth_s2]
        extra['stage3']['drop_path_rates'] = dpr[depth_s2:depth_s2 + depth_s3]
        extra['stage4']['drop_path_rates'] = dpr[depth_s2 + depth_s3:]

        upsample_cfg = extra.get('upsample', {
            'mode': 'bilinear',
            'align_corners': False
        })
        extra['upsample'] = upsample_cfg

        self.return_map = return_map

        # for partwise clustering
        nh_list = extra.get('nh_list', [1, 1, 1, 1])
        nw_list = extra.get('nw_list', [1, 1, 1, 1])
        for i in range(4-len(nh_list)):
            nh_list = [nh_list[0] * 2] + nh_list
        for i in range(4 - len(nw_list)):
            nw_list = [nw_list[0] * 2] + nw_list
        self.nh_list = nh_list
        self.nw_list = nw_list

        self.attn_type = extra.get('attn_type', 'window')

        self.ctm_with_act = extra.get('ctm_with_act', True)

        # cluster
        self.cluster_tran = extra.get('cluster_tran', [False, True, True, True])
        self.have_cluster = [False]
        input_with_cluster = [False]

        # stage1
        extra['stage1']['input_with_cluster'] = input_with_cluster

        # tran1
        self.have_cluster.append(input_with_cluster[-1])
        input_with_cluster.append(self.cluster_tran[1])

        # stage2
        extra['stage2']['input_with_cluster'] = input_with_cluster
        remerge = extra['stage2'].get('remerge', [False] * extra['stage2']['num_modules'])
        remerge = sum(remerge)
        if remerge:
            input_with_cluster = [False, True]

        # tran2
        self.have_cluster.append(input_with_cluster[-1])
        input_with_cluster.append(self.cluster_tran[2])

        # stage3
        extra['stage3']['input_with_cluster'] = input_with_cluster
        remerge = extra['stage3'].get('remerge', [False] * extra['stage3']['num_modules'])
        remerge = sum(remerge)
        if remerge:
            input_with_cluster = [False, True, True]

        # tran3
        self.have_cluster.append(input_with_cluster[-1])
        input_with_cluster.append(self.cluster_tran[3])

        # stage4
        extra['stage4']['input_with_cluster'] = input_with_cluster

        if vis:
            self.count = 0
        super().__init__(extra, in_channels, conv_cfg, norm_cfg, norm_eval,
                         with_cp, zero_init_residual)

    def _make_stage(self, layer_config, in_channels, multiscale_output=True):
        """Make stage."""
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]

        num_heads = layer_config['num_heads']
        num_window_sizes = layer_config['num_window_sizes']
        num_mlp_ratios = layer_config['num_mlp_ratios']
        drop_path_rates = layer_config['drop_path_rates']
        remerge = layer_config.get('remerge', [False] * num_modules)
        ignore_density = layer_config.get('ignore_density', [False] * num_modules)
        input_with_cluster = layer_config['input_with_cluster']

        hr_modules = []
        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            hr_modules.append(
                HRTCModule(
                    num_branches,
                    block,
                    num_blocks,
                    in_channels,
                    num_channels,
                    reset_multiscale_output,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    upsample_cfg=self.upsample_cfg,
                    num_heads=num_heads,
                    num_window_sizes=num_window_sizes,
                    num_mlp_ratios=num_mlp_ratios,
                    drop_paths=drop_path_rates[num_blocks[0] *
                                               i:num_blocks[0] * (i + 1)],
                    bilinear_upsample=self.bilinear_upsample,
                    attn_type=self.attn_type,
                    nh_list=self.nh_list,
                    nw_list=self.nw_list,
                    remerge=remerge[i],
                    ignore_density=ignore_density[i],
                    input_with_cluster=input_with_cluster,
                )
            )
            if remerge[i]:
                input_with_cluster = [False] + [True] * (num_branches - 1)

        return nn.Sequential(*hr_modules), in_channels

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        pre_stage = len(num_channels_pre_layer) - 1

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    # only change channels
                    transition_layers.append(
                        DictLayer(
                            nn.Sequential(
                                TokenConv(
                                    in_channels=num_channels_pre_layer[i],
                                    out_channels=num_channels_cur_layer[i],
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=False),
                                TokenNorm(build_norm_layer(self.norm_cfg, num_channels_cur_layer[i])),
                                nn.ReLU(inplace=True),
                            )
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                # down layers
                down_layers = CTM_partpad_dict_BN(
                    embed_dim=num_channels_pre_layer[-1],
                    dim_out=num_channels_cur_layer[i],
                    drop_rate=0,
                    sample_ratio=0.25,
                    nh_list=self.nh_list,
                    nw_list=self.nw_list,
                    level=pre_stage + 1,
                    with_act=self.ctm_with_act,
                    norm_cfg=self.norm_cfg,
                    remain_res=(self.attn_type == 'part'),
                    cluster=self.cluster_tran[num_branches_cur-1],
                    first_cluster=(not self.have_cluster[num_branches_cur-1]),
                )
                transition_layers.append(down_layers)

        return nn.ModuleList(transition_layers)

    def init_dict(self, x):
        B, C, H, W = x.shape
        device = x.device
        x = x.flatten(2).permute(0, 2, 1)
        loc_orig = get_grid_loc(B, H, W, device)
        B, N, _ = x.shape
        idx_agg = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        out_dict = {
            'x': x,
            'idx_agg': idx_agg,
            'agg_weight': agg_weight,
            'loc_orig': loc_orig,
            'map_size': [H, W]
        }
        return out_dict

    def tran2map(self, input_list):
        for i in range(len(input_list)):
            input_dict = input_list[i]
            x = input_dict['x']
            H, W = input_dict['map_size']
            idx_agg = input_dict['idx_agg']
            loc_orig = input_dict['loc_orig']
            x, _ = token2map(x, None, loc_orig, idx_agg, [H, W])
            input_list[i] = x
        return input_list

    def forward(self, x):
        """Forward function."""
        if vis:
            img = x.clone()

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.init_dict(x)

        # for debug
        # print('for debug only')
        # import matplotlib.pyplot as plt
        # IMAGENET_DEFAULT_MEAN = torch.tensor([0.485, 0.456, 0.406], device=img.device)[None, :, None, None]
        # IMAGENET_DEFAULT_STD = torch.tensor([0.229, 0.224, 0.225], device=img.device)[None, :, None, None]
        # tmp = img * IMAGENET_DEFAULT_STD + IMAGENET_DEFAULT_MEAN
        #
        # plt.subplot(1, 2, 1)
        # plt.imshow(tmp[0].permute(1, 2, 0).detach().cpu().float())
        # plt.subplot(1, 2, 2)
        # tmp = pca_feature(x['x'].detach().float())
        # tmp_map = token2map(tmp, None, x['loc_orig'], x['idx_agg'], x['map_size'])[0]
        # plt.imshow(tmp_map[0].permute(1, 2, 0).detach().cpu().float())

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

        # if vis:
        #     tmp_list = [t for t in x_list]
        #     tmp_list[-1] = tmp_list[-1][0]
        #     show_tokens_merge(img, tmp_list, self.count)
        #     self.count += 1

        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["num_branches"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        if self.return_map:
            y_list = self.tran2map(y_list)

        if vis:
            show_tokens_merge(img, x_list, self.count)
            self.count += 1
            # import matplotlib.pyplot as plt
            # plt.close()

        return y_list
