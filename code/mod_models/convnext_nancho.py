# How to train:
# python3 tools/train.py configs/kuzushiji.py


"""
python3 tools/analysis_tools/analyze_logs.py plot_curve work_dirs/hr32/20220914_155157.log.json --keys s1.loss_bbox
python3 tools/analysis_tools/analyze_logs.py plot_curve work_dirs/hr32/20220914_155157.log.json --keys loss

python3 tools/test.py configs/kuzushiji.py  work_dirs/hr32/latest.pth  --eval mAP
python3 tools/test.py configs/kuzushiji.py  work_dirs/hr32/latest.pth  --out work_dirs/hr32/test_result.pkl
python3 tools/analysis_tools/analyze_results.py configs/kuzushiji.py work_dirs/hr32/test_result.pkl work_dirs/hr32/show/

# todo:
cambiar base= cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco.py
ver como agregar special metrics

"""
#Test


# The new config inherits a base config to highlight the necessary modification
_base_ = 'convnext/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco.py'  #'cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco.py' #cascade_rcnn/cascade_rcnn_r50_caffe_fpn_1x_coco.py

# We also need to change the num_classes in head to match the dataset's annotation
# model = dict(
#     roi_head=dict(
#         _delete_=True,
#         bbox_head=dict(num_classes=2),)  #4789

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=2),
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=2),
            dict(
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

    #
    # # explicitly over-write all the `num_classes` field from default 80 to 5.
    # mask_head=dict(class_agnostic=1))
    )

# Modify dataset related settings
dataset_type = 'CustomDataset'
data_root = '/home/mauricio/Documents/Pytorch/mmdetection/data/Nancho_dataset/' #  kuzushiji_morpho2_section
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
        img_scale=[(size, size) for size in range(540, 1180 + 1, 128)], #(640, 1280 + 1, 128)
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
        type='MultiScaleFlipAug', #'MultiScaleAug',
        img_scale=[(size, size) for size in range(540, 1180 + 1, 128)],    #[(size, size) for size in range(540, 1180 + 1, 128)],#img_scale=[1.0],
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
        type='MultiScaleFlipAug',
        img_scale=[(size, size) for size in range(768, 1280 + 1, 128)],  ##[(size / 1024) for size in range(768, 1280 + 1, 128)], [0.75, 0.875, 1.0, 1.125, 1.25]
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
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'dtrain_crop.pkl', #dtrainval_crop
        img_prefix=data_root + 'train_crops1/',
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
        ann_file=data_root + 'dtest.pkl',
        img_prefix=data_root + 'test_images/',
        pipeline=test_pipeline))

# optimizer
#  max_epochs = 25
#  num_last_epochs = 1
evaluation = dict(interval=1, metric='mAP', save_best='mAP')  #save_best='mAP', dynamic_intervals=[(max_epochs - num_last_epochs, 1)]
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
# optimizer_config = dict(max_norm=35, norm_type=2)  # dict(grad_clip=None) #dict(max_norm=35, norm_type=2)
# learning policy
lr_config = dict(
    # policy='step',
    # warmup='linear',
    # warmup_iters=35, #500
    # warmup_ratio=1.0 / 3,
    step=[8, 17, 26, 35, 42],  #[8,14,20]
)
# We can set the checkpoint saving interval to reduce the storage cost
checkpoint_config = dict(interval=10)
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
total_epochs = epochs  #12
runner = dict(type='EpochBasedRunner', max_epochs=epochs)
fp16 = None
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/convnext_nancho_pre_kuzh'
# We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = './checkpoints/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220509_204200-8f07c40b.pth' # mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth
load_from = './work_dirs/convnext_nancho_kuzh_section/best_mAP_epoch_13.pth'
# resume_from = './work_dirs/nancho_pret_kuz/best_mAP_epoch_13.pth'
workflow = [('train', 1), ('val', 1)]
#  https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_Tutorial.ipynb