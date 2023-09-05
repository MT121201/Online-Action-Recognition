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

## Test the installation
```bash
./scripts/demo.sh
```

##Running test  Pipeline
```bash
python running/oad_pipeline.py
--detection ,type=str, default="/models/person/yolox_s_hdet_vga_60e_onnxonly_1280x800.onnx", help='path to detection model'
--action ,type=str, default="/models/person/action_regcornize.onnx", help='path to action model'
--s_size ,type=tuple, default=(3,3), help='size of super image'
--threshold ,type=float, default=0.5, help='threshold of action model'
--input ,type=str, help='path to input video'
--save ,type=str, default='./cache/video', help='path to save video'
--class_name ,type=str, default='/data/its/oad/triet_test/class.txt', help='class name txt file'
--device , type=str , default ="gpu", help="cpu or gpu"
```