# Export ONNX
## Install MMDeploy
This repo is testing under [MMDeploy V1.2.0](https://github.com/open-mmlab/mmdeploy/releases/tag/v1.2.0) 
```bash
git clone --recursive -b main https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy
git checkout v1.2.0
python3 tools/scripts/build_ubuntu_x64_ort.py $(nproc)
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$(pwd)/../mmdeploy-dep/onnxruntime-linux-x64-1.8.1/lib/:$LD_LIBRARY_PATH
```
## Convert model
```bash
cd mmdeploy
# convert mmpretrain model to onnxruntime model with dynamic shape
python tools/deploy.py \
    configs/mmpretrain/classification_onnxruntime_dynamic.py \
    Your model config path \
    Your model checkpoint \
    One picture for testing when deploy \
    --work-dir path to saving workdir \
    --device cpu \
    --dump-info

# example command
python tools/deploy.py \
    configs/mmpretrain/classification_onnxruntime_dynamic.py \
    ../configs/oad/repvgg_oad.py \
    /checkpoints/oad/example/trainoverfit_repvgg_oad.pth \
    /data/its/oad/data_v1_20230914/cropped_images_resized/0_62_0_2person.mp4_cropped.jpg \
    --work-dir work_dir/test/repvgg \
    --device cpu \
    --dump-info
```

- The output onnx model and test image is saved under  `mmdeploy/work_dir/`