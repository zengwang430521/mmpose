_base_ = ['_base_/datasets/coco_wholebody.py']
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(interval=10, metric='mAP', save_best='AP')

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={'relative_position_bias_table': dict(decay_mult=0.)}
    )
)

optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[170, 200])
# total_epochs = 210
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[45, 60])
total_epochs = 70


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=133,
    dataset_joints=133,
    dataset_channel=[
        list(range(133)),
    ],
    inference_channel=list(range(133)))

# model settings
fp16 = dict(loss_scale='dynamic')
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='TopDown',
    # backbone=dict(type='pvt_v2_b2', pretrained=None),
    backbone=dict(type='pvt_v2_b2',pretrained='work_dirs/pvtv2/epoch_210_backbone.pth'),
    neck=dict(
        type='AttenNeck0',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        start_level=0,
        # add_extra_convs='on_input',
        num_outs=1,
        num_heads=[4, 4, 4, 4],
        mlp_ratios=[4, 4, 4, 4],
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=256,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1, ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))


data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    bbox_file='tests/data/coco/test_coco_det_AP_H_56.json',

)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data_root = 'data/coco'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=2),
    test_dataloader=dict(samples_per_gpu=2),
    train=dict(
        type='TopDownCocoWholeBodyDataset',
        ann_file=f'tests/data/coco/test_coco_wholebody.json',
        img_prefix=f'tests/data/coco/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='TopDownCocoWholeBodyDataset',
        ann_file=f'tests/data/coco/test_coco_wholebody.json',
        img_prefix=f'tests/data/coco/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    vak=dict(
        type='TopDownCocoWholeBodyDataset',
        ann_file=f'tests/data/coco/test_coco_wholebody.json',
        img_prefix=f'tests/data/coco/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
