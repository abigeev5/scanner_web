model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=400),
        pretrained=None),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=400)))
dataset_type = 'CocoDataset'
data_root = '/content/drive/MyDrive/'
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomCrop', crop_size=(800, 600)),
    dict(
        type='Resize',
        img_scale=(800, 600),
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=[
        dict(
            type='CocoDataset',
            data_root='/content/drive/MyDrive/4х-0х/',
            ann_file='4х-0х_09_08.json',
            img_prefix='data',
            classes=('Enterobius vermicularis', ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='RandomCrop', crop_size=(800, 600)),
                dict(
                    type='Resize',
                    img_scale=(800, 600),
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            data_root=
            '/content/drive/MyDrive/Номер 2 1506380463.zip (Unzipped Files)/Номер 2 1506380463/4x/',
            ann_file='1506380463_4x_09_08.json',
            img_prefix='data',
            classes=('Enterobius vermicularis', ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='RandomCrop', crop_size=(800, 600)),
                dict(
                    type='Resize',
                    img_scale=(800, 600),
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            data_root=
            '/content/drive/MyDrive/Номер 4 1506380465.zip (Unzipped Files)/Номер 4 1506380465/4x/',
            ann_file='1506380465_4x_09_08.json',
            img_prefix='data',
            classes=('Enterobius vermicularis', ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='RandomCrop', crop_size=(800, 600)),
                dict(
                    type='Resize',
                    img_scale=(800, 600),
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            data_root=
            '/content/drive/MyDrive/Архипова 1508359041.zip (Unzipped Files)/Архипова 1508359041/4х/',
            ann_file='1508359041_4x_09_08.json',
            img_prefix='data',
            classes=('Enterobius vermicularis', ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='RandomCrop', crop_size=(800, 600)),
                dict(
                    type='Resize',
                    img_scale=(800, 600),
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            data_root=
            '/content/drive/MyDrive/Номер 9 1506380513/Номер 9 1506380513/4х-0х/',
            ann_file='1506380513_4x_09_08.json',
            img_prefix='data',
            classes=('Enterobius vermicularis', ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='RandomCrop', crop_size=(800, 600)),
                dict(
                    type='Resize',
                    img_scale=(800, 600),
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            data_root=
            '/content/drive/MyDrive/Номер _ 1509586778/Номер _ 1509586778/4х_0х/',
            ann_file='1509586778_4x_09_08.json',
            img_prefix='data',
            classes=('Enterobius vermicularis', ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='RandomCrop', crop_size=(800, 600)),
                dict(
                    type='Resize',
                    img_scale=(800, 600),
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            data_root=
            '/content/drive/MyDrive/Номер _ 1508496502/Номер _ 1508496502/4х_0/',
            ann_file='1508496502_4x_09_08.json',
            img_prefix='data',
            classes=('Enterobius vermicularis', ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='RandomCrop', crop_size=(800, 600)),
                dict(
                    type='Resize',
                    img_scale=(800, 600),
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            data_root='/content/drive/MyDrive/1508252842/',
            ann_file='1508252842_4x_manual_correction_23_08.json',
            img_prefix='data',
            classes=('Enterobius vermicularis', ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='RandomCrop', crop_size=(800, 600)),
                dict(
                    type='Resize',
                    img_scale=(800, 600),
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            data_root='/content/drive/MyDrive/1600040988/',
            ann_file='1600040988_4x_manual_correction_05_09.json',
            img_prefix='data',
            classes=('Enterobius vermicularis', ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='RandomCrop', crop_size=(800, 600)),
                dict(
                    type='Resize',
                    img_scale=(800, 600),
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            data_root='/content/drive/MyDrive/0201142-003644/',
            ann_file='0201142-003644_4x.json',
            img_prefix='data',
            classes=('Enterobius vermicularis', ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='RandomCrop', crop_size=(800, 600)),
                dict(
                    type='Resize',
                    img_scale=(800, 600),
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            data_root='/content/drive/MyDrive/1300242798/',
            ann_file='1300242798_4x.json',
            img_prefix='data',
            classes=('Enterobius vermicularis', ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='RandomCrop', crop_size=(800, 600)),
                dict(
                    type='Resize',
                    img_scale=(800, 600),
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            data_root='/content/drive/MyDrive/1509193831/',
            ann_file='1509193831_4x.json',
            img_prefix='data',
            classes=('Enterobius vermicularis', ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='RandomCrop', crop_size=(800, 600)),
                dict(
                    type='Resize',
                    img_scale=(800, 600),
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            data_root='/content/drive/MyDrive/1509358256/',
            ann_file='1509358256_4x.json',
            img_prefix='data',
            classes=('Enterobius vermicularis', ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='RandomCrop', crop_size=(800, 600)),
                dict(
                    type='Resize',
                    img_scale=(800, 600),
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            data_root='/content/drive/MyDrive/1509358270/',
            ann_file='1509358270_4x.json',
            img_prefix='data',
            classes=('Enterobius vermicularis', ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='RandomCrop', crop_size=(800, 600)),
                dict(
                    type='Resize',
                    img_scale=(800, 600),
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]),
        dict(
            type='CocoDataset',
            data_root='/content/drive/MyDrive/1509698406/',
            ann_file='1509698406_4x.json',
            img_prefix='data',
            classes=('Enterobius vermicularis', ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='RandomCrop', crop_size=(800, 600)),
                dict(
                    type='Resize',
                    img_scale=(800, 600),
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ])
    ],
    val=dict(
        type='CocoDataset',
        ann_file='1508252842_4x_for_result_sahi_full_sliced_coco.json',
        img_prefix='data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 600),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('Enterobius vermicularis', ),
        data_root='/content/drive/MyDrive/1508252842_4x_slice/'),
    test=dict(
        type='CocoDataset',
        ann_file='1509193831_4x.json',
        img_prefix='data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(800, 600),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('Enterobius vermicularis', ),
        data_root=
        '/content/drive/MyDrive/1509193831_old_dummy_for_inference_model_without_manual_correction/'
    ))
evaluation = dict(interval=2, metric='bbox', save_best='auto')
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup=None,
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=2)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
dataset_A_train = dict(
    type='CocoDataset',
    data_root='/content/drive/MyDrive/4х-0х/',
    ann_file='4х-0х_09_08.json',
    img_prefix='data',
    classes=('Enterobius vermicularis', ),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomCrop', crop_size=(800, 600)),
        dict(
            type='Resize',
            img_scale=(800, 600),
            multiscale_mode='value',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
dataset_D_train = dict(
    type='CocoDataset',
    data_root=
    '/content/drive/MyDrive/Номер 2 1506380463.zip (Unzipped Files)/Номер 2 1506380463/4x/',
    ann_file='1506380463_4x_09_08.json',
    img_prefix='data',
    classes=('Enterobius vermicularis', ),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomCrop', crop_size=(800, 600)),
        dict(
            type='Resize',
            img_scale=(800, 600),
            multiscale_mode='value',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
dataset_E_train = dict(
    type='CocoDataset',
    data_root=
    '/content/drive/MyDrive/Номер 4 1506380465.zip (Unzipped Files)/Номер 4 1506380465/4x/',
    ann_file='1506380465_4x_09_08.json',
    img_prefix='data',
    classes=('Enterobius vermicularis', ),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomCrop', crop_size=(800, 600)),
        dict(
            type='Resize',
            img_scale=(800, 600),
            multiscale_mode='value',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
dataset_F_train = dict(
    type='CocoDataset',
    data_root=
    '/content/drive/MyDrive/Архипова 1508359041.zip (Unzipped Files)/Архипова 1508359041/4х/',
    ann_file='1508359041_4x_09_08.json',
    img_prefix='data',
    classes=('Enterobius vermicularis', ),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomCrop', crop_size=(800, 600)),
        dict(
            type='Resize',
            img_scale=(800, 600),
            multiscale_mode='value',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
dataset_G_train = dict(
    type='CocoDataset',
    data_root=
    '/content/drive/MyDrive/Номер 9 1506380513/Номер 9 1506380513/4х-0х/',
    ann_file='1506380513_4x_09_08.json',
    img_prefix='data',
    classes=('Enterobius vermicularis', ),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomCrop', crop_size=(800, 600)),
        dict(
            type='Resize',
            img_scale=(800, 600),
            multiscale_mode='value',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
dataset_H_train = dict(
    type='CocoDataset',
    data_root=
    '/content/drive/MyDrive/Номер _ 1509586778/Номер _ 1509586778/4х_0х/',
    ann_file='1509586778_4x_09_08.json',
    img_prefix='data',
    classes=('Enterobius vermicularis', ),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomCrop', crop_size=(800, 600)),
        dict(
            type='Resize',
            img_scale=(800, 600),
            multiscale_mode='value',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
dataset_I_train = dict(
    type='CocoDataset',
    data_root=
    '/content/drive/MyDrive/Номер _ 1508496502/Номер _ 1508496502/4х_0/',
    ann_file='1508496502_4x_09_08.json',
    img_prefix='data',
    classes=('Enterobius vermicularis', ),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomCrop', crop_size=(800, 600)),
        dict(
            type='Resize',
            img_scale=(800, 600),
            multiscale_mode='value',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
dataset_J_train = dict(
    type='CocoDataset',
    data_root='/content/drive/MyDrive/1508252842/',
    ann_file='1508252842_4x_manual_correction_23_08.json',
    img_prefix='data',
    classes=('Enterobius vermicularis', ),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomCrop', crop_size=(800, 600)),
        dict(
            type='Resize',
            img_scale=(800, 600),
            multiscale_mode='value',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
dataset_K_train = dict(
    type='CocoDataset',
    data_root='/content/drive/MyDrive/1600040988/',
    ann_file='1600040988_4x_manual_correction_05_09.json',
    img_prefix='data',
    classes=('Enterobius vermicularis', ),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomCrop', crop_size=(800, 600)),
        dict(
            type='Resize',
            img_scale=(800, 600),
            multiscale_mode='value',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
dataset_L_train = dict(
    type='CocoDataset',
    data_root='/content/drive/MyDrive/0201142-003644/',
    ann_file='0201142-003644_4x.json',
    img_prefix='data',
    classes=('Enterobius vermicularis', ),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomCrop', crop_size=(800, 600)),
        dict(
            type='Resize',
            img_scale=(800, 600),
            multiscale_mode='value',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
dataset_M_train = dict(
    type='CocoDataset',
    data_root='/content/drive/MyDrive/1300242798/',
    ann_file='1300242798_4x.json',
    img_prefix='data',
    classes=('Enterobius vermicularis', ),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomCrop', crop_size=(800, 600)),
        dict(
            type='Resize',
            img_scale=(800, 600),
            multiscale_mode='value',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
dataset_N_train = dict(
    type='CocoDataset',
    data_root='/content/drive/MyDrive/1509193831/',
    ann_file='1509193831_4x.json',
    img_prefix='data',
    classes=('Enterobius vermicularis', ),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomCrop', crop_size=(800, 600)),
        dict(
            type='Resize',
            img_scale=(800, 600),
            multiscale_mode='value',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
dataset_O_train = dict(
    type='CocoDataset',
    data_root='/content/drive/MyDrive/1509358256/',
    ann_file='1509358256_4x.json',
    img_prefix='data',
    classes=('Enterobius vermicularis', ),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomCrop', crop_size=(800, 600)),
        dict(
            type='Resize',
            img_scale=(800, 600),
            multiscale_mode='value',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
dataset_P_train = dict(
    type='CocoDataset',
    data_root='/content/drive/MyDrive/1509358270/',
    ann_file='1509358270_4x.json',
    img_prefix='data',
    classes=('Enterobius vermicularis', ),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomCrop', crop_size=(800, 600)),
        dict(
            type='Resize',
            img_scale=(800, 600),
            multiscale_mode='value',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
dataset_Q_train = dict(
    type='CocoDataset',
    data_root='/content/drive/MyDrive/1509698406/',
    ann_file='1509698406_4x.json',
    img_prefix='data',
    classes=('Enterobius vermicularis', ),
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomCrop', crop_size=(800, 600)),
        dict(
            type='Resize',
            img_scale=(800, 600),
            multiscale_mode='value',
            keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ])
work_dir = './tutorial_exps'
seed = 0
gpu_ids = range(0, 1)
device = 'cuda'

