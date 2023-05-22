_base_ = 'mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
                    type='Shared4Conv1FCBBoxHead',
                    conv_out_channels=256,
                    num_classes=2
                    ),
        bbox_roi_extractor=dict(
            type='GenericRoIExtractor',
            aggregation='sum',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
            pre_cfg=dict(
                type='ConvModule',
                in_channels=256,
                out_channels=256,
                kernel_size=5,
                padding=2,
                inplace=False,
            ),
            post_cfg=dict(
                type='GeneralizedAttention',
                in_channels=256,
                spatial_range=-1,
                num_heads=6,
                attention_type='0100',
                kv_stride=2))))

dataset_type = 'CustomDataset'
data_root = '/home/mauricio/Documents/Pytorch/mmdetection/mmdetection_mau/data/Nancho_dataset/' #  kuzushiji_morpho2_section
classes = ('background', 'c',)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

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
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(
        type='Resize',
        img_scale=[(576, 1024), (544, 1024),      #(480, 1024), (512, 1024),
                    (608, 1024), (640, 1024), (672, 1024), (704, 1024),
                    (736, 1024), (768, 1024), (800, 1024),
                   (924, 1024), (1024, 1024)],
        # img_scale=[(480, 1024), (512, 1024), (544, 1024), (576, 1024),
        #                    (608, 1024), (640, 1024)],  ##[(size, size) for size in range(540, 1180 + 1, 128)], #(640, 1280 + 1, 128)
        keep_ratio=True,
        multiscale_mode='value'),
    dict(
        type='Albu',
        transforms=albu_train_transforms
    ),
    dict(type='RandomFlip', flip_ratio=float(0)),  #commented
    #  dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='RandomCrop', crop_size=(640, 640)),  #fpg
    # dict(type='Pad', size=(640, 640)),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',  #'MultiScaleAug',
        # img_scale=(1333, 800),
        img_scale=[(576, 1024), (544, 1024),      #(480, 1024), (512, 1024),
                    (608, 1024), (640, 1024), (672, 1024), (704, 1024),
                    (736, 1024), (768, 1024), (800, 1024),
                   (924, 1024), (1024, 1024)],
        # img_scale=[(640, 640)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            # dict(type='Pad', size=(640, 640)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(576, 1024), (544, 1024),      #(480, 1024), (512, 1024),
                    (608, 1024), (640, 1024), (672, 1024), (704, 1024),
                    (736, 1024), (768, 1024), (800, 1024),
                   (924, 1024), (1024, 1024)],  # [(size, size) for size in range(768, 1280 + 1, 128)],  ##[(size / 1024) for size in range(768, 1280 + 1, 128)], [0.75, 0.875, 1.0, 1.125, 1.25]
        # img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            # dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    # workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'dtrain_crop.pkl', #dtrainval_crop  #modified
        img_prefix=data_root + 'train_crops1/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
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
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22, 30, 42])
# lr_config = dict(step=[16, 22, 30, 42])
runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=10)  #10
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
fp16 = None
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/ground_true_nancho'
# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = './checkpoints/mask_rcnn_r50_fpn_groie_1x_coco_20200604_211715-50d90c74.pth' # mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth
# load_from = './work_dirs/convnext_nancho_kuzh_section/best_mAP_epoch_13.pth'
# resume_from = './work_dirs/convnext_nancho_pre_kuzh_PAFPN/epoch_10.pth'
workflow = [('train', 1), ('val', 1)]