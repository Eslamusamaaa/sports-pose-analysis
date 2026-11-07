# ST-GCN fine-tuned for 4 swimming strokes on SwimXYZ (CC-BY 4.0).
# Starts from the NTU RGB+D 60 2D-COCO checkpoint (backbone kept, head resized).
#
# Windows are 60 contiguous frames at 30 fps (2 s) - the web app's swimming
# timeline sends exactly this, so UniformSampleFrames(clip_len=60) is an
# identity at inference and never time-compresses the stroke.
_base_ = r'C:/Users/Laptop World/Downloads/Body_Pose-20260924T150348Z-1-001/Body_Pose/mmaction2-main/configs/_base_/default_runtime.py'

custom_imports = dict(imports=['swim_transforms'], allow_failed_imports=False)

model = dict(
    type='RecognizerGCN',
    backbone=dict(type='STGCN', graph_cfg=dict(layout='coco', mode='stgcn_spatial')),
    cls_head=dict(type='GCNHead', num_classes=4, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = r'D:/Users/karate-training/datasets/swimxyz/swimxyz_coco17_split.pkl'
CLIP = 60

train_pipeline = [
    dict(type='SwimSkeletonAug'),
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=CLIP),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=CLIP, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline,
                 split='train'))
val_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline,
                 split='val', test_mode=True))
test_dataloader = val_dataloader

val_evaluator = [dict(type='AccMetric', metric_list=('top_k_accuracy', 'mean_class_accuracy'))]
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=15, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=1,
         convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', eta_min=0, T_max=14, by_epoch=True, begin=1, end=15,
         convert_to_iter_based=True)
]
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005, nesterov=True))

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=2, save_best='acc/top1', rule='greater'),
    logger=dict(interval=100))

load_from = r'C:/Users/Laptop World/Downloads/Body_Pose-20260924T150348Z-1-001/Body_Pose/karate-ai-web/models/mmaction/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth'
work_dir = r'D:/Users/karate-training/swim/work_dir'
auto_scale_lr = dict(enable=False, base_batch_size=128)
