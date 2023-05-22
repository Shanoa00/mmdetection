# python3 tools/misc/browse_dataset.py configs/fpg_nancho_50e_coco.py
#  python3 tools/analysis_tools/analyze_logs.py plot_curve work_dirs/convnext_nancho_fpg/20221110_133421.log.json  --keys loss_bbox
_base_ = 'fpg/mask_rcnn_r50_fpn_crop640_50e_coco.py'  #faster_rcnn_r50_fpn_crop640_50e_coco.py

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    neck=dict(
        type='FPG',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        inter_channels=256,
        num_outs=5,
        stack_times=9,
        paths=['bu'] * 9,
        same_down_trans=None,
        same_up_trans=dict(
            type='conv',
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            inplace=False,
            order=('act', 'conv', 'norm')),
        across_lateral_trans=dict(
            type='conv',
            kernel_size=1,
            norm_cfg=norm_cfg,
            inplace=False,
            order=('act', 'conv', 'norm')),
        across_down_trans=dict(
            type='interpolation_conv',
            mode='nearest',
            kernel_size=3,
            norm_cfg=norm_cfg,
            order=('act', 'conv', 'norm'),
            inplace=False),
        across_up_trans=None,
        across_skip_trans=dict(
            type='conv',
            kernel_size=1,
            norm_cfg=norm_cfg,
            inplace=False,
            order=('act', 'conv', 'norm')),
        output_trans=dict(
            type='last_conv',
            kernel_size=3,
            order=('act', 'conv', 'norm'),
            inplace=False),
        norm_cfg=norm_cfg,
        skip_inds=[(0, 1, 2, 3), (0, 1, 2), (0, 1), (0, ), ()]),
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            num_classes=2,
        )
    )
)
# Modify dataset related settings
dataset_type = 'CustomDataset'
data_root = '/home/mauricio/Documents/Pytorch/mmdetection/mmdetection_mau/data/Nancho_dataset/' #  kuzushiji_morpho2_section
classes = ('background', 'c',)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# train_initial_size = 1024  #2000->Good results #max=1024
# crop_min_max_height = (400, 533)  # (560,693) batsize:12 -> (550, 683), original(520, 653)
# crop_width = 512  ##595, batsize:12 ->595  original (575)
# crop_height = 384  #467, #batsize:12 ->467   original (447)
albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='HueSaturationValue',
                hue_shift_limit=7,
                sat_shift_limit=10,
                val_shift_limit=10,
                )
        ]
    ),
    # dict(type='LongestMaxSize', max_size=train_initial_size),  #added
    # dict(
    #     type='RandomSizedCrop',  ##added
    #     min_max_height=crop_min_max_height,
    #     width=crop_width,
    #     height=crop_height,
    #     w2h_ratio=crop_width / crop_height,
    # ),
    dict(
        type='RandomBrightnessContrast'
    ),
    dict(
        type='RandomGamma'
    ),
    dict(
        type='RandomContrast',
        limit=0.05,
        p=0.75
    ),
    dict(
        type='RandomBrightness',
        limit=0.05,
        p=0.75
    ),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='Resize',
        img_scale=[(640, 640)],
        ratio_range=(0.8, 1.2),
        keep_ratio=True,
        # multiscale_mode='value'
        ),
    # dict(
    #     type='Albu',
    #     transforms=albu_train_transforms
    # ),
    dict(type='RandomFlip', flip_ratio=float(0)),  #commented
    dict(type='RandomCrop', crop_size=(640, 640)),
    #  dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    dict(type='Pad', size=(640, 640)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',  #'MultiScaleAug',
        img_scale=[(640, 640)],
        # ratio_range=(0.8, 1.2),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=64),
            dict(type='Pad', size=(640, 640)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(640, 640)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size=(640, 640)),
            dict(type='Pad', size_divisor=64),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'dtrain_crop.pkl',  #dtrainval_crop  #modified
        img_prefix=data_root + 'train_crops1/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        # ann_file=data_root + 'dtest.pkl',
        # img_prefix=data_root + 'test_images/',
        # pipeline=test_pipeline),
        ann_file=data_root + 'dval.pkl',   #modified
        img_prefix=data_root + 'train_images/',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'dtest.pkl',
        img_prefix=data_root + 'test_images/',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='mAP', save_best='mAP')  #   ,save_best='mAP', dynamic_intervals=[(max_epochs - num_last_epochs, 1)]
# optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
#                  paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
#                                                  'relative_position_bias_table': dict(decay_mult=0.),
#                                                  'norm': dict(decay_mult=0.)}))

# learning policy
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(norm_decay_mult=0, bypass_duplicate=True)
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    step=[20, 30, 40]
)
# We can set the checkpoint saving interval to reduce the storage cost
checkpoint_config = dict(interval=10)  #10
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
epochs = 50
runner = dict(type='EpochBasedRunner', max_epochs=epochs)
fp16 = None
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/convnext_nancho_fpg'
# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = './checkpoints/mask_rcnn_r50_fpg_crop640_50e_coco_20220311_011857-233b8334.pth' # mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth
# load_from = './work_dirs/convnext_nancho_kuzh_section/best_mAP_epoch_13.pth'
# resume_from = './work_dirs/convnext_nancho_pre_kuzh_PAFPN/epoch_10.pth'
workflow = [('train', 1), ('val', 1)]
#  https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_Tutorial.ipynb
