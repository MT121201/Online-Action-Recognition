_base_ = "../mmpretrain/repvgg/repvgg-A0_8xb32_in1k.py"

custom_imports = dict(
    imports=[
        'tni_pretrain.datasets.transforms.load_super_image',
        'tni_pretrain.datasets.transforms.super_image_from_list',
        'tni_pretrain.datasets.super_image_imagenet',
        'tni_pretrain.engine.hooks.superimage_visualization_hook',
    ],
    allow_failed_imports=False)

# classes
class_name = ('call', 'read', 'purchase', 'chat')

num_classes = len(class_name)
metainfo = dict(classes=class_name)
# Load pretrained model from github
load_from = "https://download.openmmlab.com/mmclassification/v0/repvgg/repvgg-A0_8xb32_in1k_20221213-60ae8e23.pth"

# Change head numclasses and loss weight
model = dict(
    head=dict(num_classes=num_classes,
              loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))

# Change num classes in preprocessor
data_preprocessor = dict(
    num_classes=num_classes,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

# dataset
dataset_type = 'SuperImage'

# for debugging, training only with single image
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='RandomResizedCrop',
#         scale=224,
#         backend='pillow',
#         interpolation='bicubic'),
#     dict(type='RandomFlip', prob=0.5, direction='horizontal'),
#     dict(
#         type='RandAugment',
#         policies='timm_increasing',
#         num_policies=2,
#         total_level=10,
#         magnitude_level=9,
#         magnitude_std=0.5,
#         hparams=dict(
#             pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
#     dict(
#         type='RandomErasing',
#         erase_prob=0.25,
#         mode='rand',
#         min_area_ratio=0.02,
#         max_area_ratio=1 / 3,
#         fill_color=bgr_mean,
#         fill_std=bgr_std),
#     dict(type='PackInputs')
# ]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
#     dict(type='CenterCrop', crop_size=224),
#     dict(type='PackInputs'),
# ]

train_pipeline = [
    dict(type='LoadSuperImageFromList', croped_image_size=(100, 128)),
    dict(type='ColorJitter', brightness=0.4,
         contrast=0.4, saturation=0.4, hue=0.4),
    dict(type='GaussianBlur', prob=0.5, radius=2),
    dict(type='Lighting', eigval=[0.2175, 0.0188, 0.0045],
         eigvec=[[-0.5675, 0.7192, 0.4009],
                 [-0.5808, -0.0045, -0.8140],
                 [-0.5836, -0.6948, 0.4203]],),
    dict(type='Rotate', angle=20, prob=0.5),
    dict(backend='pillow', scale=224, type='RandomResizedCrop'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomGrayscale', prob=0.2, keep_channels=True),
    dict(type='Resize', scale=(224, 224)),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadSuperImageFromList', croped_image_size=(100, 128)),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

dataset_1_train = dict(
    type=dataset_type,
    data_root='/data/its/oad/data_v1_20230914',
    metainfo=metainfo,
    ann_file='annotations/train.json',
    data_prefix='cropped_images_resized',
    pipeline=train_pipeline,
    with_label=True,
    super_image_grid=(3, 3),
    frame_per_group=2,)


# repeat dataset
dataset_1_train_repeat = dict(
    type='RepeatDataset',
    _delete_=True,
    times=50,
    dataset=dataset_1_train
)

# # Concat dataset
# dataset_concat = dict(
#     type='ConcatDataset',
#     _delete_=True,
#     datasets=[dataset_1_train_repeat])

train_dataloader = dict(
    num_workers=6,
    batch_size=256,
    dataset=dataset_1_train_repeat
)

# Val dataloaders
dataset_1_val = dict(
    type=dataset_type,
    data_root='/data/its/oad/data_v1_20230914',
    metainfo=metainfo,
    ann_file='annotations/test.json',
    data_prefix='cropped_images_resized',
    with_label=True,
    pipeline=test_pipeline,
    frame_per_group=2
)


# Apply concat dataset to val
dataset_val = dict(
    type='ConcatDataset',
    _delete_=True,
    datasets=[dataset_1_val])

val_dataloader = dict(
    batch_size=64,
    dataset=dataset_val
)

# accuracy
val_evaluator = dict(type='Accuracy', topk=(1, 2))

# OR Precision, Recall, F1-score
# val_evaluator = dict(
#     _delete_=True,
#     type='SingleLabelMetric',
#     average=None, # Print out classwise
#     # average='macro' # average
# )

test_dataloader = val_dataloader
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=2)
default_hooks = dict(checkpoint=dict(type='CheckpointHook',
                     interval=10, max_keep_ckpts=2, save_best='auto'),
                     visualization=dict(type='SuperImageVisualizationHook'))

# fp16 training help you x2 batch size
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale=512.0)
