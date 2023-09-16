# Infer Video
## Prepare config
```
Check the config inside "configs/infer_oad/repvva0_oad_with_persondet_infer.py"
```
## Running
```bash
# TODO: For test we can use the video path in "Running command demo" below or this "/data/its/oad/video/combined_video.mp4"
python running/oad_pipeline.py
    --input path to the input video \
    --save saving path, if not give will save to ./cache/video with name as input +"_output.mp4" \
    --use_detector if set, using Detector and ByteTrack to detect and track object, otherwise using CenterCrop coordinates
```
### Demo
#### "configs/infer_oad/repvva0_oad_with_persondet_infer.py"
Check the config before running inference
#### Running command demo
```bash
python running/oad_pipeline.py \
    --input /data/its/oad/video/self_record.mp4 \
    --use_detector # Without this will use CenterCrop coordinates

This will be saved in ./cache/video/demo_output.mp4, because not given --save
```
Explain about CenterCrop in config file
```bash
centercrop = dict(
    crop_size = (H,W)
)
(H,W) = (int,int) e.g.(108,108) will crop H=108 and W=108
(H,W) = (int,float) e.g.(108, 0.5) or vice versa will crop H=108 and W=0.5*frame_width
(H,W) = (int,-1) e.g.(108, -1) or vice versa will crop H=108 and W=frame_width
(H,W) = (int,0) e.g.(108, 0) or vice versa will crop H=108 and Keep Aspect Ratio
(H,W) = (int) e.g.(108) will square crop H=108 and W=108
(H,W) = (float) e.g.(0.5) will crop by Frame Ratio: H=0.5*frame_height and W=0.5*frame_width
```
