# Prepare training dataset from video
## Instruction video
- Watch [this video](https://drive.google.com/file/d/1jm3JPmuhkjFzd5-oZxjd6MJ-tuAGC_y1/view?usp=sharing) to know how to label, convert, and modify config for training new data.

## Labeling
```bash
Using CVAT label object with track mode, export as CVAT Video (xml file)
```
## Cropping from video
```bash
python tools/oad/crop_video.py \
    --video path to video \
    --xml path to CVAT ann file \
    --crop_save path to save croped images folder
```
## Resize croped frame to improve dataloader speed in training
```bash
python tools/oad/pretraining_resize.py \
    -input input folder to resize, the folder keep croped images \
    -save output to save resized images, if None will be input + "_resized" \
    -H Height \
    -W Width \
    --remove if set, will remove the cropped images folder after resize it
if 1 of H or W is not given, will resize to the given dimension and keep ratio
```
## Making JSON annotation file from xml
```bash
python tools/oad/xml_json_ann.py \
    --xml path to CVAT ann file \
    --save path to saving folder \
    --video path to video \
    --split, if given will auto split to train/test annotation files
```

## Full create dataset
```bash
python tools/oad/prepare_dataset.py \
    --video path to your video \
    --xml path to CVAT ann file \
    --save path to saving dataset \
    --H Height \
    --W Width \
    --remove if set, will remove the cropped images folder after resize it
if 1 of H or W is not given, will resize to the given dimension and keep ratio
```
