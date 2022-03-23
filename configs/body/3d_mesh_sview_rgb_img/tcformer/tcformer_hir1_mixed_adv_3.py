log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(interval=5, metric='joint_error', save_best='p-mpjpe')

use_adversarial_train = True

optimizer = dict(
    generator=dict(
        type='AdamW',
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01),
    discriminator=dict(type='Adam', lr=1e-4))
optimizer_config = None

lr_config = dict(policy='Fixed', by_epoch=False)

total_epochs = 100
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

img_res = 224

# model settings
model = dict(
    type='ParametricMesh',
    backbone=dict(type='tcformer_small', pretrained='models/checkpoint_tcformer.pth'),
    neck=dict(type='HirAttNeck1'),
    mesh_head=dict(
        type='HMRMeshHead',
        in_channels=512,
        smpl_mean_params='models/smpl/smpl_mean_params.npz',
    ),
    disc=dict(),
    smpl=dict(
        type='SMPL',
        smpl_path='models/smpl',
        joints_regressor='models/smpl/jr_spin_train_24.npy',
        joints_regressor_eval='models/smpl/jr_spin_eval_24.npy',
    ),
    train_cfg=dict(disc_step=1),
    test_cfg=dict(),
    loss_mesh=dict(
        type='MeshLoss2',
        joints_2d_loss_weight=1,
        joints_3d_loss_weight=2,
        vertex_loss_weight=2,
        smpl_pose_loss_weight=2,
        smpl_beta_loss_weight=0.002,
        focal_length=5000,
        img_res=img_res),
    loss_gan=dict(
        type='GANLoss',
        gan_type='lsgan',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=0.001))

data_cfg = dict(
    image_size=[img_res, img_res],
    iuv_size=[img_res // 4, img_res // 4],
    num_joints=24,
    use_IUV=False,
    uv_type='BF')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MeshRandomChannelNoise', noise_factor=0.4),
    dict(type='MeshRandomFlip', flip_prob=0.5),
    dict(type='MeshGetRandomScaleRotation', rot_factor=30, scale_factor=0.25),
    dict(type='MeshAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=[
            'img', 'joints_2d', 'joints_2d_visible', 'joints_3d',
            'joints_3d_visible', 'pose', 'beta', 'has_smpl'
        ],
        meta_keys=['image_file', 'center', 'scale', 'rotation']),
]

train_adv_pipeline = [dict(type='Collect', keys=['mosh_theta'], meta_keys=[])]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MeshAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=[
            'img', 'joints_2d', 'joints_2d_visible', 'joints_3d',
            'joints_3d_visible', 'pose', 'beta', 'has_smpl', 'gender'
        ],
        meta_keys=['image_file', 'center', 'scale', 'rotation']),
]

test_pipeline = val_pipeline

len2d_eft = [1000, 14810, 9428, 28344]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='MeshAdversarialDataset',
        train_dataset=dict(
            type='MeshMixDataset',
            configs=[
                dict(
                    ann_file='data/mesh_annotation_files/h36m_train_new.npz',
                    img_prefix='data/h36m_train',
                    data_cfg=data_cfg,
                    pipeline=train_pipeline),
                dict(
                    ann_file='data/mesh_annotation_files/'
                             'mpi_inf_3dhp_train.npz',
                    img_prefix='data/mpi_inf_3dhp',
                    data_cfg=data_cfg,
                    pipeline=train_pipeline),
                dict(
                    ann_file='data/mesh_annotation_files/'
                             '3dpw_train.npz',
                    img_prefix='data/3DPW',
                    data_cfg=data_cfg,
                    pipeline=train_pipeline),
                dict(
                    ann_file='data/mesh_annotation_files/'
                             'lsp_dataset_original_train.npz',
                    img_prefix='data/lsp_dataset_original',
                    data_cfg=data_cfg,
                    pipeline=train_pipeline),
                dict(
                    ann_file='data/mesh_annotation_files/mpii_train_eft.npz',
                    img_prefix='data/mpii',
                    data_cfg=data_cfg,
                    pipeline=train_pipeline),
                dict(
                    ann_file='data/mesh_annotation_files/hr-lspet_train_eft.npz',
                    img_prefix='data/hr-lspet',
                    data_cfg=data_cfg,
                    pipeline=train_pipeline),
                dict(
                    ann_file='data/mesh_annotation_files/coco_2014_train_eft.npz',
                    img_prefix='data/coco',
                    data_cfg=data_cfg,
                    pipeline=train_pipeline)
            ],
            partition=[0.3, 0.1, 0.2] + [0.4 * l / sum(len2d_eft) for l in len2d_eft]),
        adversarial_dataset=dict(
            type='MoshDataset',
            ann_file='data/mesh_annotation_files/CMU_mosh.npz',
            pipeline=train_adv_pipeline),
    ),
    test=dict(
        type='MeshH36MDataset',
        ann_file='data/mesh_annotation_files/h36m_valid_protocol2.npz',
        img_prefix='data/Human3.6M',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
    ),
    val=dict(
        type='Mesh3DPWDataset',
        ann_file='data/mesh_annotation_files/3dpw_test.npz',
        img_prefix='data/3DPW',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
    ),
)