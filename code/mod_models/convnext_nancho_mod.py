# How to train:
# python3 tools/train.py configs/convnext_nancho_mod.py


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

# norm_cfg = dict(type='BN', requires_grad=True) #nas_fpn (best performance) https://github.com/open-mmlab/mmdetection/tree/master/configs/nas_fpn
norm_cfg = dict(type='BN', requires_grad=True)  # FPG
model = dict(
    # neck=dict(
    #     type='PAFPN'),
    # neck=dict(    #nas_fpn (bn)(second best performance) https://github.com/open-mmlab/mmdetection/tree/master/configs/nas_fpn
    #     relu_before_extra_convs=True,
    #     no_norm_on_lateral=True,
    #     norm_cfg=norm_cfg),
    # neck=dict(   #carafe https://github.com/open-mmlab/mmdetection/blob/master/configs/carafe/mask_rcnn_r50_fpn_carafe_1x_coco.py
    #     type='FPN_CARAFE',
    #     in_channels=[96, 192, 384, 768],
    #     out_channels=256,
    #     num_outs=5,
    #     start_level=0,
    #     end_level=-1,
    #     norm_cfg=None,
    #     act_cfg=None,
    #     order=('conv', 'norm', 'act'),
    #     upsample_cfg=dict(
    #         type='carafe',
    #         up_kernel=5,
    #         up_group=1,
    #         encoder_kernel=3,
    #         encoder_dilation=1,
    #         compressed_channels=64)),
    # ------------
    # neck=dict(  #  FPG BEST PERFORMANCE  https://github.com/open-mmlab/mmdetection/blob/master/configs/fpg/retinanet_r50_fpg_crop640_50e_coco.py
    #     type='FPN',
    #     in_channels=[96, 192, 384, 768], #[256, 512, 1024, 2048],
    #     out_channels=256,
    #     norm_cfg=norm_cfg,
    #     num_outs=5),
    neck=dict(
        type='FPG',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        inter_channels=255,
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
        skip_inds=[(0, 1, 2, 3), (0, 1, 2), (0, 1), (0,), ()]),
    # neck=[  # DyHead (NOT WORKING!!) https://github.com/open-mmlab/mmdetection/blob/master/configs/dyhead/atss_r50_fpn_dyhead_1x_coco.py
    #         dict(
    #             type='FPN',
    #             in_channels=[96, 192, 384, 768],  #[256, 512, 1024, 2048],
    #             out_channels=256,
    #             start_level=1,
    #             add_extra_convs='on_output',
    #             num_outs=5),
    #         dict(type='DyHead', in_channels=256, out_channels=256, num_blocks=6)
    #     ],
    # neck=dict(type='HRFPN', in_channels=[96, 192, 384, 768], out_channels=256), #mal performance, idea del 1st place
    roi_head=dict(
        # bbox_roi_extractor=dict(  #groie  https://github.com/open-mmlab/mmdetection/blob/master/configs/groie/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco.py
        #             type='GenericRoIExtractor',
        #             aggregation='sum',
        #             roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
        #             out_channels=256,
        #             featmap_strides=[4, 8, 16, 32],
        #             pre_cfg=dict(
        #                 type='ConvModule',
        #                 in_channels=256,
        #                 out_channels=256,
        #                 kernel_size=5,
        #                 padding=2,
        #                 inplace=False,
        #             ),
        #             post_cfg=dict(
        #                 type='GeneralizedAttention',
        #                 in_channels=256,
        #                 spatial_range=-1,
        #                 num_heads=6,
        #                 attention_type='0100',
        #                 kv_stride=2)),
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

    #
    # # explicitly over-write all the `num_classes` field from default 80 to 5.
    # mask_head=dict(class_agnostic=1))
    )

# Modify dataset related settings
dataset_type = 'CustomDataset'
data_root = '/home/mauricio/Documents/Pytorch/mmdetection/mmdetection_mau/data/Nancho_dataset/' #  kuzushiji_morpho2_section
classes = ('background', 'c',)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_initial_size = 1024  #2000->Good results #max=1024
crop_min_max_height = (400, 533)  # (560,693) batsize:12 -> (550, 683), original(520, 653)
crop_width = 512  ##595, batsize:12 ->595  original (575)
crop_height = 384  #467, #batsize:12 ->467   original (447)
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
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',  #'MultiScaleAug',
        img_scale=[(576, 1024), (544, 1024),      #(480, 1024), (512, 1024),
                    (608, 1024), (640, 1024), (672, 1024), (704, 1024),
                    (736, 1024), (768, 1024), (800, 1024),
                   (924, 1024), (1024, 1024)],
        # img_scale=[(480, 1024), (512, 1024), (544, 1024), (576, 1024),
        #                    (608, 1024), (640, 1024), (672, 1024), (704, 1024),
        #                    (736, 1024), (768, 1024), (800, 1024)],
        # img_scale=[(480, 1024), (512, 1024), (544, 1024), (576, 1024),
        #                    (608, 1024), (640, 1024)],  #[(size, size) for size in range(540, 1180 + 1, 128)],   #[(size, size) for size in range(540, 1180 + 1, 128)],    #[(size, size) for size in range(540, 1180 + 1, 128)],#img_scale=[1.0],
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
        img_scale=[(576, 1024), (544, 1024),  # (480, 1024), (512, 1024),
                   (608, 1024), (640, 1024), (672, 1024), (704, 1024),
                   (736, 1024), (768, 1024), (800, 1024),
                   (924, 1024), (1024, 1024)],  # [(size, size) for size in range(768, 1280 + 1, 128)],  ##[(size / 1024) for size in range(768, 1280 + 1, 128)], [0.75, 0.875, 1.0, 1.125, 1.25]
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

optimizer = dict(  ## original optimizer
    _delete_=True,
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=0.0004,   #0.0004
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.8,  #0.7
        'decay_type': 'layer_wise',
        'num_layers': 6
    })

# # 1 GPU * 4 samples_per_gpu * 4 cumulative_iters
# # to simulate 4 GPUs * 4 samples_per_gpu
# optimizer_config = dict(cumulative_iters=4)
# lr_config = dict(warmup_iters=2000)  # 500 * cumulative_iters

# optimizer_config = dict(max_norm=35, norm_type=2)  # dict(grad_clip=None) #dict(max_norm=35, norm_type=2)
# learning policy
lr_config = dict(
    warmup_iters=100,
    # policy='step',
    # warmup='linear',
    # warmup_ratio=1.0 / 3,
    min_lr=3.294e-06,  #added
    step=[8, 17, 26, 35, 42],  #[8, 17, 26, 35, 42], [8,14,20]
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
total_epochs = epochs  #12
runner = dict(type='EpochBasedRunner', max_epochs=epochs)
fp16 = None
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/convnext_nancho_necks'
# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = './checkpoints/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220509_204200-8f07c40b.pth' # mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth
# load_from = './work_dirs/convnext_nancho_kuzh_section/best_mAP_epoch_13.pth'
# resume_from = './work_dirs/convnext_nancho_pre_kuzh_PAFPN/epoch_10.pth'
workflow = [('train', 1), ('val', 1)]
#  https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_Tutorial.ipynb