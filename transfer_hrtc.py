import torch
from mmpose.models.backbones.tc_module.hr_tcformer import HRTCFormer
from mmpose.models import build_posenet
import copy

channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

norm_cfg = dict(type='SyncBN', requires_grad=True)

model_cfg = dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='HRTCFormer',
        in_channels=3,
        norm_cfg=norm_cfg,
        return_map=True,
        extra=dict(
            nh_list=[8, 4, 2],
            nw_list=[8, 4, 2],
            drop_path_rate=0.1,
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(2,),
                num_channels=(64,),
                num_heads=[2],
                num_mlp_ratios=[4]),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='TCWINBLOCK',
                num_blocks=(2, 2),
                num_channels=(32, 64),
                num_heads=[1, 2],
                num_mlp_ratios=[4, 4],
                num_window_sizes=[7, 7]),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='TCWINBLOCK',
                num_blocks=(2, 2, 2),
                num_channels=(32, 64, 128),
                num_heads=[1, 2, 4],
                num_mlp_ratios=[4, 4, 4],
                num_window_sizes=[7, 7, 7]),
            stage4=dict(
                num_modules=2,
                num_branches=4,
                block='TCWINBLOCK',
                num_blocks=(2, 2, 2, 2),
                num_channels=(32, 64, 128, 256),
                num_heads=[1, 2, 4, 8],
                num_mlp_ratios=[4, 4, 4, 4],
                num_window_sizes=[7, 7, 7, 7],
                multiscale_output=True,
            )
        )),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=32,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        # norm_cfg=norm_cfg,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11)

)

# model_cfg = dict(
#     type='TopDown',
#     pretrained=None,
#     backbone=dict(
#         type='HRTCFormer',
#         in_channels=3,
#         norm_cfg=norm_cfg,
#         return_map=True,
#         extra=dict(
#             nh_list=[8, 4, 2],
#             nw_list=[8, 4, 2],
#             drop_path_rate=0.1,
#             stage1=dict(
#                 num_modules=1,
#                 num_branches=1,
#                 block='BOTTLENECK',
#                 num_blocks=(2,),
#                 num_channels=(64,),
#                 num_heads=[2],
#                 num_mlp_ratios=[4]),
#             stage2=dict(
#                 num_modules=1,
#                 num_branches=2,
#                 block='TCWINBLOCK',
#                 num_blocks=(2, 2),
#                 num_channels=(78, 156),
#                 num_heads=[1, 2],
#                 num_mlp_ratios=[4, 4],
#                 num_window_sizes=[7, 7]),
#             stage3=dict(
#                 num_modules=4,
#                 num_branches=3,
#                 block='TCWINBLOCK',
#                 num_blocks=(2, 2, 2),
#                 num_channels=(78, 156, 312),
#                 num_heads=[1, 2, 4],
#                 num_mlp_ratios=[4, 4, 4],
#                 num_window_sizes=[7, 7, 7]),
#             stage4=dict(
#                 num_modules=2,
#                 num_branches=4,
#                 block='TCWINBLOCK',
#                 num_blocks=(2, 2, 2, 2),
#                 num_channels=(78, 156, 312, 624),
#                 num_heads=[1, 2, 4, 8],
#                 num_mlp_ratios=[4, 4, 4, 4],
#                 num_window_sizes=[7, 7, 7, 7])
#         )),
#     keypoint_head=dict(
#         type='TopdownHeatmapSimpleHead',
#         in_channels=32,
#         out_channels=channel_cfg['num_output_channels'],
#         num_deconv_layers=0,
#         # norm_cfg=norm_cfg,
#         extra=dict(final_conv_kernel=1, ),
#         loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
#     train_cfg=dict(),
#     test_cfg=dict(
#         flip_test=True,
#         post_process='default',
#         shift_heatmap=True,
#         modulate_kernel=11)
#
# )



device = torch.device('cuda')

model = build_posenet(model_cfg)

model_dict = model.state_dict()


# src_file = 'models/hrt_small_coco_256x192.pth'
src_file = 'models/hrt_small.pth'
# src_file = 'models/hrt_base.pth'

out_file = src_file.replace('hrt', 'hrtcformer')

src_dict = torch.load(src_file)
src_dict = src_dict['state_dict'] if 'state_dict' in src_dict.keys() else src_dict['model']



src_left = copy.copy(src_dict)
model_left = copy.copy(model_dict)
out_dict = {}

# remove relative bias
for src_key in src_dict.keys():
    if 'relative_position' in src_key:
        src_left.pop(src_key)


for model_key in model_dict.keys():
    src_key = model_key.replace('backbone.', '')

    if src_key in src_dict.keys():
        out_dict[model_key] = src_dict[src_key]
        src_left.pop(src_key)
        model_left.pop(model_key)

    elif 'skip' in model_key:
        out_dict[model_key] = model_dict[model_key] * 0
        model_left.pop(model_key)

    elif '.transition' in model_key:
        # transition without cluster
        if '.layer.' in model_key:
            src_key = src_key.replace('.layer.', '.')
            src_key = src_key.replace('.norm.', '.')
            if src_key in src_dict:
                out_dict[model_key] = src_dict[src_key]
                src_left.pop(src_key)
                model_left.pop(model_key)
            elif 'skip.weight' in model_key:
                out_dict[model_key] = model_dict[model_key] * 0
                model_left.pop(model_key)
            else:
                # print(model_key)
                print(src_key)

        # transition with cluster
        else:
            src_key = src_key.replace('.conv.', '.0.0.')
            src_key = src_key.replace('.norm.', '.0.1.')
            if src_key in src_dict:
                out_dict[model_key] = src_dict[src_key]
                src_left.pop(src_key)
                model_left.pop(model_key)
            elif 'skip' in model_key:
                out_dict[model_key] = model_dict[model_key] * 0
                model_left.pop(model_key)
            else:
                print(src_key)

    # attn
    elif '.attn.' in model_key:
        src_key = src_key.replace('.attn.', '.attn.attn.')
        src_key = src_key.replace('.q.', '.q_proj.')
        src_key = src_key.replace('.proj.', '.out_proj.')
        if src_key in src_dict:
            out_dict[model_key] = src_dict[src_key]
            src_left.pop(src_key)
            model_left.pop(model_key)
        elif 'skip' in model_key:
            out_dict[model_key] = model_dict[model_key] * 0
            model_left.pop(model_key)
        elif '.kv.' in model_key:
            src_k = src_key.replace('.kv.', '.k_proj.')
            src_v = src_key.replace('.kv.', '.v_proj.')
            tmp_k = src_dict[src_k]
            tmp_v = src_dict[src_v]
            tmp_kv = torch.cat([tmp_k, tmp_v], dim=0)
            out_dict[model_key] = tmp_kv
            model_left.pop(model_key)
            src_left.pop(src_k)
            src_left.pop(src_v)
        else:
            print(src_key)

    # fuse layer
    elif 'fuse_layers' in model_key:
        src_key = src_key.replace('.fuse_layers.fuse_layers.', '.fuse_layers.')
        src_key = src_key.replace('.layer.', '.')
        src_key = src_key.replace('.dw_conv.', '.0.')
        src_key = src_key.replace('.norm1.', '.1.')
        src_key = src_key.replace('.conv.', '.2.')
        src_key = src_key.replace('.norm2.', '.3.')
        src_key = src_key.replace('.norm.', '.')

        if src_key in src_dict:
            if src_dict[src_key].shape == model_dict[model_key].shape:
                out_dict[model_key] = src_dict[src_key]
                src_left.pop(src_key)
                model_left.pop(model_key)
            else:
                out_dict[model_key] = src_dict[src_key].squeeze(-1).squeeze(-1)
                src_left.pop(src_key)
                model_left.pop(model_key)
        else:
            print(src_key)
    else:
        print(src_key)

print(len(out_dict))
print(len(src_left))
print(len(model_left))

print('Unloaded parameters:')
for key in model_left.keys():
    if '.bias' in key:
        out_dict[key] = model_left[key] * 0
    elif '.conf' in key:
        out_dict[key] = model_left[key] * 0
    else:
        out_dict[key] = model_left[key]
    print(key)

# try to load
tmp = model.load_state_dict(out_dict)
print(tmp)

torch.save(out_dict, out_file)
t = 0
