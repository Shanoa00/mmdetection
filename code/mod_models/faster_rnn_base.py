_base_ = [
    '_base_/models/faster_rcnn_r50_fpn.py',
    '_base_/datasets/coco_detection.py',
    '_base_/schedules/schedule_1x.py', '_base_/default_runtime.py'
]
# _base_ = 'mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
# norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
#     backbone=dict(
#         norm_cfg=norm_cfg,
#         init_cfg=dict(
#             type='Pretrained',
#             checkpoint='open-mmlab://detectron/resnet50_gn')),
#     neck=dict(norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            num_classes=2,
            # norm_cfg=norm_cfg
            ),
        ))
# img_norm_cfg = dict(
#     mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# Modify dataset related settings
dataset_type = 'CustomDataset'
dataset_name= 'kuzushiji/'  #kuzushiji, #HanDataset,  #Nancho_dataset,
data_root = '/workspace/mmdetection_mau/data/'+ dataset_name
classes = ('background', 'c',)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#https://github.com/open-mmlab/mmdetection/blob/master/configs/albu_example/mask_rcnn_r50_fpn_albu_1x_coco.py
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
        # img_scale=[(540, 540), (668, 668), (796, 796), (924, 924),
        #            (1052, 1052), (1180, 1180)],
        img_scale=[(576, 576), (544, 544),  # sss(480, 1024), (512, 1024),
                   (608, 608), (640, 640),
                   (736, 736), (768, 768), (800, 800),
                   (924, 924), (1024, 1024)],
        # img_scale=[(576, 1024), (544, 1024),  # (480, 1024), (512, 1024),
        #            (608, 1024), (640, 1024), (672, 1024), (704, 1024),
        #            (736, 1024), (768, 1024), (800, 1024),
        #            (924, 1024), (1024, 1024)],
        # img_scale=[(480, 1024), (512, 1024), (544, 1024), (576, 1024),
        #                    (608, 1024), (640, 1024)],  ##[(size, size) for size in range(540, 1180 + 1, 128)], #(640, 1280 + 1, 128)
        keep_ratio=True,
        multiscale_mode='value'),
    dict(
        type='Albu',
        transforms=albu_train_transforms
    ),
     dict(type='RandomFlip', flip_ratio=float(0)),
    #  dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',  # 'MultiScaleAug',
        img_scale=[(768, 768), (896, 896), (1024, 1024), (1152, 1152),
                           (1280, 1280)],
        # img_scale=[(540, 540), (668, 668), (796, 796), (924, 924),
        #            (1052, 1052), (1180, 1180)],
        # img_scale=[(576, 1024), (544, 1024),  # (480, 1024), (512, 1024),
        #            (608, 1024), (640, 1024), (672, 1024), (704, 1024),
        #            (736, 1024), (768, 1024), (800, 1024),
        #            (924, 1024), (1024, 1024)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',  # 'MultiScaleAug',
        img_scale=[(768, 768), (896, 896), (1024, 1024), (1152, 1152),
                           (1280, 1280)],
        # img_scale=[(540, 540), (668, 668), (796, 796), (924, 924),
        #            (1052, 1052), (1180, 1180)],
        # img_scale=[(576, 1024), (544, 1024),  # (480, 1024), (512, 1024),
        #            (608, 1024), (640, 1024), (672, 1024), (704, 1024),
        #            (736, 1024), (768, 1024), (800, 1024),
        #            (924, 1024), (1024, 1024)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'dtrain.pkl',  #dtrain_crop2.pkl, 'dtrain_crop1240.pkl', 0dtrain_crop (last used in scale folder)
        img_prefix=data_root + 'train_images/',  #train_crops2, 'train_crops1240/', 0train_crops1
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'dval.pkl',
        img_prefix=data_root + 'train_images/',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'dtest.pkl',  #0dtest
        img_prefix=data_root + 'test_images/',
        pipeline=test_pipeline))

# optimizer
#  max_epochs = 25
#  num_last_epochs = 1
evaluation = dict(interval=1, metric='mAP', save_best='mAP')  #save_best='mAP', dynamic_intervals=[(max_epochs - num_last_epochs, 1)]
# optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
#                  paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
#                                                  'relative_position_bias_table': dict(decay_mult=0.),
#                                                  'norm': dict(decay_mult=0.)}))
# optimizer_config = dict(max_norm=35, norm_type=2)  # dict(grad_clip=None) #dict(max_norm=35, norm_type=2)
# learning policy
# lr_config = dict(
#     warmup_iters=100,
#     # policy='step',
#     # warmup='linear',
#     # warmup_ratio=1.0 / 3,
#     min_lr=3.294e-06,  # added
#     step=[8, 17, 26, 35, 42],  #[8,14,20]
# )
# We can set the checkpoint saving interval to reduce the storage cost
checkpoint_config = dict(interval=6)  #25
# yapf:disable
log_config = dict(
    interval=600, #50
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        # dict(type='MMDetWandbHook',
        #      init_kwargs={'project': 'DefFPN'},
        #      interval=50,
        #      log_checkpoint=True,
        #      log_checkpoint_metadata=True,
        #      num_eval_images=20)
    ])
# yapf:enable
# runtime settings
epochs = 31  #50  31
total_epochs = epochs  #12
runner = dict(type='EpochBasedRunner', max_epochs=epochs)
fp16 = None
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/' + dataset_name+'faster_kuzushiji'  #swin_nancho_deformable2, swin_nancho_deformable2/scales, S03/swin_t
# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = './checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
#load_from = './checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth' #
# load_from = './work_dirs/swin_nancho_kuzh_section/best_mAP_epoch_19.pth'
# resume_from = './work_dirs/nancho_pret_kuz/best_mAP_epoch_13.pth'
workflow = [('train', 1), ('val', 1)]
#  https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_Tutorial.ipynb