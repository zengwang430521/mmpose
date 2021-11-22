import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

from .pvt_v2 import (Block, DropPath, DWConv, OverlapPatchEmbed, trunc_normal_, _cfg, get_root_logger, load_checkpoint)

from .utils_mine import (
    get_grid_loc,
    gumble_top_k, index_points,
    map2token_agg_fast_nearest,  # map2token_agg_mat_nearest, map2token_agg_sparse_nearest
    show_tokens_merge, show_conf_merge, merge_tokens, merge_tokens_agg_dist, token2map_agg_mat,
    tokenconv_sparse,
    # farthest_point_sample
)

from .utils_mine import token_cluster_topk as token_cluster_density
# from utils_mine import token2map_agg_sparse as token2map_agg_mat
from ..builder import BACKBONES
from .utils_mine import DPC_flops, token2map_flops, map2token_flops, downup_flops, sra_flops

vis = False
# vis = True

from .pvt_v2_3h2_density_fix import MyMlp, MyDWConv, MyAttention, MyBlock



'''
do not select tokens, merge tokens. weight NOT clamp, conf do not clamp
merge feature, but not merge locs, reserve all locs.
inherit weights when map2token, which can regarded as tokens merge
farthest_point_sample DOWN, N_grid = 0, feature distance merge
token2map nearest + skip token conv (this must be used together.)
try to make it faster

dist_assign, No ada dc
fix a bug in gathering
TOPK, for ablation

'''





# from partialconv2d import PartialConv2d
class DownLayer(nn.Module):
    """ Down sample
    """

    def __init__(self, sample_ratio, embed_dim, dim_out, drop_rate, down_block,
                 k=3, dist_assign=True, ada_dc=False, use_conf=False, conf_scale=0.25, conf_density=False):
        super().__init__()
        # self.sample_num = sample_num
        self.sample_ratio = sample_ratio
        self.dim_out = dim_out

        self.block = down_block
        # self.pos_drop = nn.Dropout(p=drop_rate)
        # self.gumble_sigmoid = GumbelSigmoid()
        # temperature of confidence weight
        self.register_buffer('T', torch.tensor(1.0, dtype=torch.float))
        self.T_min = 1
        self.T_decay = 0.9998
        self.conv = nn.Conv2d(embed_dim, dim_out, kernel_size=3, stride=2, padding=1)
        self.conv_skip = nn.Linear(embed_dim, dim_out, bias=False)
        # self.conv = PartialConv2d(embed_dim, self.block.dim_out, kernel_size=3, stride=1, padding=1)
        self.norm = nn.LayerNorm(self.dim_out)
        self.conf = nn.Linear(self.dim_out, 1)

        # for density clustering
        self.k = k
        self.dist_assign = dist_assign
        self.ada_dc = ada_dc
        self.use_conf = use_conf
        self.conf_scale = conf_scale
        self.conf_density = conf_density

    def forward(self, x, pos_orig, idx_agg, agg_weight, H, W, N_grid, grid_merge=False):
        # x, mask = token2map(x, pos, [H, W], 1, 2, return_mask=True)
        # x = self.conv(x, mask)

        B, N, C = x.shape
        N0 = idx_agg.shape[1]
        if N0 == N and N == H * W:
            x_map = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        else:
            x_map, _ = token2map_agg_mat(x, None, pos_orig, idx_agg, [H, W])

        x_map = self.conv(x_map)
        x = map2token_agg_fast_nearest(x_map, N, pos_orig, idx_agg, agg_weight) + self.conv_skip(x)
        x = self.norm(x)
        conf = self.conf(x)
        weight = conf.exp()

        if grid_merge:
            mean_weight = weight.reshape(B, 1, H, W)
            mean_weight = F.avg_pool2d(mean_weight, kernel_size=2)
            mean_weight = F.interpolate(mean_weight, [H, W], mode='nearest')
            mean_weight = mean_weight.reshape(B, H*W, 1)
            norm_weight = weight / (mean_weight + 1e-6)

            x_down = x * norm_weight
            x_down = x_down.reshape(B, H, W, -1).permute(0, 3, 1, 2)
            x_down = F.avg_pool2d(x_down, kernel_size=2)
            x_down = x_down.flatten(2).permute(0, 2, 1)

            weight_t = norm_weight / 4

            _, _, H, W = x_map.shape
            idx_agg_down = torch.arange(H*W, device=x.device).reshape(1, 1, H, W)
            idx_agg_down = F.interpolate(idx_agg_down.float(), [H*2, W*2], mode='nearest').long()
            idx_agg_down = idx_agg_down.reshape(-1)[None, :].repeat([B, 1])

        else:
            _, _, H, W = x_map.shape
            B, N, C = x.shape
            sample_num = max(math.ceil(N * self.sample_ratio) - N_grid, 0)
            if sample_num < N_grid:
                sample_num = N_grid

            sr_ratio = self.block.attn.sr_ratio
            x_down, idx_agg_down, weight_t = token_cluster_density(
                x, sample_num, idx_agg, weight, True, conf,
                k=self.k, dist_assign=self.dist_assign, ada_dc=self.ada_dc,
                use_conf=self.use_conf, conf_scale=self.conf_scale, conf_density=self.conf_density,
                # loc_orig=pos_orig, map_size=[H // sr_ratio, W // sr_ratio]
            )

        agg_weight_down = agg_weight * weight_t
        agg_weight_down = agg_weight_down / agg_weight_down.max(dim=1, keepdim=True)[0]

        x_down = self.block(x_down, idx_agg_down, agg_weight_down, pos_orig,
                            x, idx_agg, agg_weight, H, W, conf_source=conf)

        if vis:
            show_conf_merge(conf, None, pos_orig, idx_agg)
        return x_down, idx_agg_down, agg_weight_down


class MyPVT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False,
                 k=3, dist_assign=True, ada_dc=False, use_conf=False, conf_scale=0.25, conf_density=False,
                 pretrained=None
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.depths = depths
        self.sample_ratio = 0.25
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(1):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        for i in range(1, num_stages):
            down_layers = DownLayer(sample_ratio=0.25, embed_dim=embed_dims[i-1], dim_out=embed_dims[i],
                                    drop_rate=drop_rate,
                                    down_block=MyBlock(
                                        dim=embed_dims[i], num_heads=num_heads[i],
                                        mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur],
                                        norm_layer=norm_layer, sr_ratio=sr_ratios[i], linear=linear),
                                    k=k, dist_assign=dist_assign, ada_dc=ada_dc, use_conf=use_conf,
                                    conf_scale=conf_scale, conf_density=conf_density
                                    )
            block = nn.ModuleList([MyBlock(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(1, depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"down_layers{i}", down_layers)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

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

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        if vis:
            img = x
        outs = []

        # stage 1
        i = 0
        patch_embed = getattr(self, f"patch_embed{i + 1}")
        block = getattr(self, f"block{i + 1}")
        norm = getattr(self, f"norm{i + 1}")
        x, H, W = patch_embed(x)
        for blk in block:
            x = blk(x, H, W)
        x = norm(x)

        B, N, _ = x.shape
        device = x.device
        N_grid = 0
        idx_agg = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        loc_orig = get_grid_loc(B, H, W, x.device)

        outs.append((x, None, [H, W], loc_orig, idx_agg, agg_weight))

        for i in range(1, self.num_stages):
            down_layers = getattr(self, f"down_layers{i}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, idx_agg, agg_weight = down_layers(x, loc_orig, idx_agg, agg_weight, H, W, N_grid)  # down sample
            H, W = H // 2, W // 2

            for j, blk in enumerate(block):
                x = blk(x, idx_agg, agg_weight, loc_orig, x, idx_agg, agg_weight, H, W, conf_source=None)

            x = norm(x)
            outs.append((x, None, [H, W], loc_orig, idx_agg, agg_weight))

        if vis:
            show_tokens_merge(img, outs, N_grid)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


    def get_extra_flops(self, H, W):
        flops = 0
        h, w = H // 4, W // 4
        N0 = h * w
        N = N0
        for stage in range(4):
            depth, sr, dim = self.depths[stage], self.sr_ratios[stage], self.embed_dims[stage]
            mlp_r = self.mlp_ratios[stage]
            dim_up = self.embed_dims[stage-1]

            if stage > 0:
                # cluster flops
                flops += DPC_flops(N, dim)
                flops += map2token_flops(N0, dim_up) + token2map_flops(N0, dim)
                N = N * self.sample_ratio
                h, w = h // 2, w // 2

            # attn flops
            flops += sra_flops(h, w, sr, dim) * depth

            if stage > 0:
                # map, token flops
                flops += (map2token_flops(N0, dim) + map2token_flops(N0, dim * mlp_r) + token2map_flops(N0, dim * mlp_r)) * depth


        return flops


@BACKBONES.register_module()
class mypvt3h2_density0ftopk_small(MyPVT):
    def __init__(self, **kwargs):
        super().__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            k=5, dist_assign=True, ada_dc=False, use_conf=False, conf_scale=0,
            **kwargs)

