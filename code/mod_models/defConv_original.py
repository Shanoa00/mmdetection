_base_ = 'dcn/cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco.py'

norm_cfg = dict(type='GN', num_groups=32, requires_grad=True) #nas_fpn (best performance) https://github.com/open-mmlab/mmdetection/tree/master/configs/nas_fpn

model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        # type='FPN',
        type='DefNeck',
        def_mode='All',  # 'lateral', 'All', 'TopBottom'
        in_channels=[256, 512, 1024, 2048],  #[256, 512, 1024, 2048],
        out_channels=256,
        relu_before_extra_convs=True,
        no_norm_on_lateral=True,
        norm_cfg=norm_cfg,
        num_outs=5),
    roi_head=dict(
        bbox_head=[
            dict(
                norm_cfg=norm_cfg,  # FPG
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=2),
            dict(
                norm_cfg=norm_cfg,  # FPG
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=2),
            dict(
                norm_cfg=norm_cfg,  # FPG
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=2)
        ]),
    train_cfg=dict(
            rpn_proposal=dict(
                nms=dict(type='nms', iou_threshold=0.25))),
    test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.25),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.25),
                max_per_img=1000))
)
# Modify dataset related settings
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
        img_scale=[(768, 768), (896, 896), (1024, 1024), (1152, 1152),
                   (1280, 1280)],
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
        img_scale=[(768, 768), (896, 896), (1024, 1024), (1152, 1152),
                   (1280, 1280)],
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
    samples_per_gpu=1,
    workers_per_gpu=8,
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

lr_config = dict(
    warmup_iters=100,
    # policy='step',
    # warmup='linear',
    # warmup_ratio=1.0 / 3,
    min_lr=3.294e-06,  # added
    step=[8, 17, 26, 35, 42],  #[8,14,20]
)
evaluation = dict(interval=1, metric='mAP', save_best='mAP')  #   ,save_best='mAP', dynamic_intervals=[(max_epochs - num_last_epochs, 1)]
# optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
#                  paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
#                                                  'relative_position_bias_table': dict(decay_mult=0.),
#                                                  'norm': dict(decay_mult=0.)}))
checkpoint_config = dict(interval=25)  #10
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
# total_epochs = epochs  #12
runner = dict(type='EpochBasedRunner', max_epochs=epochs)
fp16 = None
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/defConv_original'
# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = './checkpoints/mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200203-4d9ad43b.pth' # mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth
# load_from = './work_dirs/convnext_nancho_kuzh_section/best_mAP_epoch_13.pth'
# resume_from = './work_dirs/convnext_nancho_pre_kuzh_PAFPN/epoch_10.pth'
workflow = [('train', 1), ('val', 1)]