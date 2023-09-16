## Installation
### Install necessary libs using Conda
```bash
conda create --name <user>_tni_oad python=3.8 -y
conda activate <user>_tni_oad

# install pytorch
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# install mmengine, mmcv, mmpretrain
pip install -U openmim
mim install mmengine==0.8.4
mim install mmcv==2.0.1
mim install mmpretrain==1.0.0

# install this repo
python setup.py develop
```
### Install ByteTrack
```bash
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
python -m pip install -r requirements.txt
python3 setup.py develop
python -m pip install Cython
python -m pip install cython_bbox pycocotools
python -m pip install numpy==1.23.1
# Bytetrack will default replace ONNXRUNTIME-GPU by ONNXRUNTIME 1.8
pip uninstall onnxruntime
python -m pip install onnxruntime-gpu==1.15.1
```

### Test ONNXRUNTIME-GPU
```bash
python
import onnxruntime as rt
rt.get_device()
#>> "GPU"

# If this raise ERROR uninstall onnxruntime-gpu and install again
pip uninstall onnxruntime-gpu
python -m pip install onnxruntime-gpu==1.15.1
```
