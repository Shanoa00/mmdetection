# How to train:
# python3 tools/train.py configs/kuzushiji.py
"""
python3 tools/analysis_tools/analyze_logs.py plot_curve work_dirs/hr32/20220914_155157.log.json --keys s1.loss_bbox
python3 tools/analysis_tools/analyze_logs.py plot_curve work_dirs/hr32/20220914_155157.log.json --keys loss

python3 tools/test.py configs/kuzushiji.py  work_dirs/hr32/latest.pth  --eval mAP --eval-options iou_thr=0.5
python3 tools/test.py configs/kuzushiji.py  work_dirs/hr32/latest.pth  --out work_dirs/hr32/test_result.pkl
python3 tools/analysis_tools/analyze_results.py configs/kuzushiji.py work_dirs/hr32/test_result.pkl work_dirs/hr32/show/
"""

# custom_imports = dict(
#     imports=['mmdet.models.necks.fpg.py'],
#     allow_failed_imports=False)

# The new config inherits a base config to highlight the necessary modification
_base_ = 'swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'  #'cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco.py' #cascade_rcnn/cascade_rcnn_r50_caffe_fpn_1x_coco.py

# norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    backbone=dict(
        # depths=[2, 2, 6, 2],  #[4, 6, 2],  #[2, 2, 6, 2] #no pretrain models available nas worse performance
        # num_heads=[3, 6, 12, 24],  #[3, 6, 12],  #, 24]
        # window_size=3,   #7 No afecta cambiarlo
        # out_indices=(0, 1, 2, 3),  #(0, 1, 2),  #(0, 1, 2, 3)
        # init_cfg=None,
        ),
    neck=dict(  #  FPG BEST PERFORMANCE  https://github.com/open-mmlab/mmdetection/blob/master/configs/fpg/retinanet_r50_fpg_crop640_50e_coco.py
        type='DefNeck',  #FPN, DefNeck
        def_mode='TopBottom',  # 'lateral', 'All', 'TopBottom'
        in_channels=[96, 192, 384, 768],  #[96, 192, 384, 768],  [256, 512, 1024, 2048],
        out_channels=256,
        # end_level=-1,  #-1 , 3
        # num_outs=5, #5,   4
        relu_before_extra_convs=True,
        no_norm_on_lateral=True,
        # norm_cfg=norm_cfg
        ),
    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4],  #reduces from 8 ->4
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]  #, 64]
            ),),
    roi_head=dict(
        # bbox_roi_extractor=dict(
        #         type='SingleRoIExtractor',
        #         roi_layer=dict(
        #             _delete_=True,
        #             type='DeformRoIPoolPack',
        #             output_size=7,
        #             output_channels=256),
        #         out_channels=256,
        #         featmap_strides=[4, 8, 16, 32]),
    # roi_head=dict(
    #     bbox_roi_extractor=dict(  #groie  https://github.com/open-mmlab/mmdetection/blob/master/configs/groie/mask_rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco.py
    #                 type='GenericRoIExtractor',
    #                 aggregation='sum',
    #                 roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2), #output_size==7
    #                 out_channels=256,
    #                 featmap_strides=[4, 8, 16, 32],
    #                 pre_cfg=dict(
    #                     type='ConvModule',
    #                     in_channels=256,
    #                     out_channels=256,
    #                     kernel_size=4,
    #                     padding=4,
    #                     inplace=False,
    #                 )),
    #                 post_cfg=dict(
    #                     type='GeneralizedAttention',
    #                     in_channels=256,
    #                     spatial_range=-1,
    #                     num_heads=6,
    #                     attention_type='0100',
    #                     kv_stride=2)),
        bbox_head=dict(
                # norm_cfg=norm_cfg,
                num_classes=2)),
    train_cfg=dict(
            rpn_proposal=dict(
                # nms_pre=3000,  #changed for S05 2000 -> 2500 just need to comment,  2500-> 3000
                # nms=dict(type='nms', iou_threshold=0.25)
            )),
    test_cfg=dict(
            rpn=dict(
                # nms_pre=2000,  #changed for S05 1000 -> 1500,  1500-> 2000
                # max_per_img=1500,  # 1000 -> 1500
                # nms=dict(type='nms', iou_threshold=0.25),
                min_bbox_size=0),
            rcnn=dict(
                nms=dict(type='nms', iou_threshold=0.25),
                max_per_img=1000))   #1000 -> 1500
    )

# Modify dataset related settings
dataset_type = 'CustomDataset'
data_root = '/workspace/mmdetection_mau/data/Nancho_dataset/'  #S03_Detection&Recognition,  #Nancho_dataset,  kuzushiji_morpho2_section
classes = ('background', 'c',) #data_root+'classes.txt' #('background', 'c',)
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
        img_scale=[(576, 576), (544, 544),  # (480, 1024), (512, 1024),
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
        ann_file=data_root + 'dtrain_crop1240.pkl',  #'dtrain_crop1240.pkl', dtrain_crop
        img_prefix=data_root + '0train_crops1/',  #'train_crops1240/', train_crops2
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + '0dval.pkl',
        img_prefix=data_root + 'train_images/',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + '0dtest.pkl',
        img_prefix=data_root + 'test_images/',
        pipeline=test_pipeline))

# optimizer
#  max_epochs = 25
#  num_last_epochs = 1
evaluation = dict(interval=1, metric='mAP', save_best='mAP')  #save_best='mAP', dynamic_intervals=[(max_epochs - num_last_epochs, 1)]
optimizer = dict(_delete_=True, type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.05, #0.001
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
# optimizer_config = dict(max_norm=35, norm_type=2)  # dict(grad_clip=None) #dict(max_norm=35, norm_type=2)
# learning policy
lr_config = dict(
    warmup_iters=100,
    # policy='step',
    # warmup='linear',
    # warmup_ratio=1.0 / 3,
    min_lr=3.294e-04,  # added 3.294e-06
    step=[8, 17, 26, 35, 42],  #[8,14,20]
)
# We can set the checkpoint saving interval to reduce the storage cost
checkpoint_config = dict(interval=25)
# yapf:disable
log_config = dict(
    interval=25,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
epochs = 20
total_epochs = epochs  #12
runner = dict(type='EpochBasedRunner', max_epochs=epochs)
fp16 = None
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/Nancho_deff'  #swin_nancho_deformable2, swin_nancho_deformable, S03/swin_t
# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = './checkpoints/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_20210906_131725-bacf6f7b.pth' #
# load_from = './work_dirs/swin_nancho_kuzh_section/best_mAP_epoch_19.pth'
# resume_from = './work_dirs/nancho_pret_kuz/best_mAP_epoch_13.pth'
workflow = [('train', 1), ('val', 1)]
#  https://github.com/open-mmlab/mmdetection/blob/master/demo/MMDet_Tutorial.ipynb