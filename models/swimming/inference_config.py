# Inference-only config for the swimming ST-GCN (4 strokes, SwimXYZ).
# Training config (with augmentation + data paths) lives in
# D:/Users/karate-training/swim/stgcn_swim4.py; this copy has no custom
# imports so the web app's worker can load it on its own.
default_scope = 'mmaction'

model = dict(
    type='RecognizerGCN',
    backbone=dict(type='STGCN', graph_cfg=dict(layout='coco', mode='stgcn_spatial')),
    cls_head=dict(type='GCNHead', num_classes=4, in_channels=256))

test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=60, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]
