### 1. Train the model
```bash
export CUDA_VISIBLE_DEVICES=0
echo "train classification model"

WORKDIR="experiments/repvgg_oad"

# config file
CFG="configs/oad/repvgg_oad.py"

# train model
mim train mmpretrain $CFG --work-dir $WORKDIR
```
### 2. Run evaluation
```bash
export CUDA_VISIBLE_DEVICES=0

# your config file
CFG="configs/oad/repvgg_oad.py"

# the checkpoint
CHECKPOINT="path/to/ckpt"

WORKDIR="experiments/repvgg_oad"
SHOWDIR=visualize

# run test
mim test mmpretrain $CFG --checkpoint $CHECKPOINT --work-dir $WORKDIR --show-dir $SHOWDIR
```
