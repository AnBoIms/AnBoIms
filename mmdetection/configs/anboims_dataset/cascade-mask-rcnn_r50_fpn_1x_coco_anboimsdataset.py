_base_ = '../cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=1,
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
                num_classes=1,
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
                num_classes=1,
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
        mask_head=dict(num_classes=1)
    )
)

data_root = "C:/Users/Woojiho/Desktop/mmdetection/tools/data/anboims_polygon_dataset/"
metainfo = {
    'classes': ('IDCard', ),
    'palette': [
        (220, 20, 60),
    ]
}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/annotation_coco.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val/annotation_coco.json',
        data_prefix=dict(img='val/')))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'val/annotation_coco.json')
test_evaluator = val_evaluator

evaluation = dict(interval=1, metric=['bbox', 'segm'])
