_base_ = '../htc/htc-without-semantic_r50_fpn_1x_coco.py'

model = dict(
    roi_head=dict(
        _delete_=True,
        type='HybridTaskCascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1.0, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=2,  # Custom dataset has only one class
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]
                ),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)
            ),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]
                ),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)
            ),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]
                ),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
                ),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)
            )
        ],
        mask_head=[
            dict(
                type='HTCMaskHead',
                num_classes=2,
                in_channels=256,
                conv_out_channels=256,
                loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
            ),
            dict(
                type='HTCMaskHead',
                num_classes=2,
                in_channels=256,
                conv_out_channels=256,
                loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
            ),
            dict(
                type='HTCMaskHead',
                num_classes=2,
                in_channels=256,
                conv_out_channels=256,
                loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
            )
        ]
    )
)

data_root = "C:/Users/Woojiho/Desktop/mmdetection/tools/data/anboims_polygon_dataset/"
metainfo = {
    'classes': ('IDCard', ),
    'palette': [
        (220, 20, 60),
    ]
}

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),  # with_seg 제거
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/annotation_coco.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/annotation_coco.json',
        data_prefix=dict(img='val/'),
        pipeline=train_pipeline
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'val/annotation_coco.json')
test_evaluator = val_evaluator

evaluation = dict(interval=1, metric=['bbox', 'segm'])
