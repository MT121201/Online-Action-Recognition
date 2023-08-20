# Online-Action-Detection

## Installation
```bash
conda create --name <user>_tni_oad python=3.8 -y
conda activate <user>_tni_oad

# install pytorch
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# install mmengine, mmcv, mmpretrain, mmdet, mmpose
pip install -U openmim
mim install mmengine==0.8.4
mim install mmcv==2.0.1
mim install mmdet==3.1.0
mim install mmpose==1.1.0

# install mmaction2
mim install mmaction2==1.1.0

# install this repo
python setup.py develop
```
