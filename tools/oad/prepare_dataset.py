from crop_video import VideoProcessor
from xml_json_ann import DataConverter
from pretraining_resize import ImageProcessor
import argparse
import os

def argphase():
    parser = argparse.ArgumentParser(description='Make dataset from video and CVAT annotation.')
    parser.add_argument('--video', type=str, help='Path to video.')
    parser.add_argument('--xml', type=str, help='Path to annotations.')
    parser.add_argument('--save', type=str, help='Path to folder saving dataset.')
    parser.add_argument('--H', type=int, default=None, help='H to resize.')
    parser.add_argument('--W', type=int, default=None, help='W to resize.')
    parser.add_argument('--remove', action='store_true', help='Remove cropped images folder after have resized cropped images.')
    args = parser.parse_args()
    return args

def main():
    args = argphase()
    video_path = args.video
    xml_path = args.xml
    save_path = args.save
    H = args.H
    W = args.W
    remove = args.remove

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(video_path):
        raise Exception("Video file not found")
    if not os.path.exists(xml_path):
        raise Exception("XML file not found")
    
    crop_save_path = os.path.join(save_path, 'cropped_images')
    if not os.path.exists(crop_save_path):
        os.makedirs(crop_save_path)
    video_processor = VideoProcessor(video_path=video_path, 
                                     xml_path=xml_path, 
                                     output_folder=crop_save_path)
    video_processor.process_video()

    resize_save_path = os.path.join(save_path, 'resized_images')
    if not os.path.exists(resize_save_path):
        os.makedirs(resize_save_path)
    image_processor = ImageProcessor(input_folder=crop_save_path, 
                                     save_folder=resize_save_path, 
                                     H=H, 
                                     W=W, 
                                     remove=remove)
    image_processor.processing_image()

    xml_json_save_path = os.path.join(save_path, 'annotations')
    if not os.path.exists(xml_json_save_path):
        os.makedirs(xml_json_save_path)
    xml_json_ann = DataConverter(xml_path=xml_path, 
                                 video_path=video_path, 
                                 save_path=xml_json_save_path,
                                 split=True)
    xml_json_ann.make_json_ann()

if __name__ == '__main__':
    main()

## test COMMAND: python tools/oad/prepare_dataset.py --video /data/its/oad/video/self_record.mp4 --xml /data/its/oad/triet_test/cvat_ann/self_record1.xml --save /data/its/oad/test --H 128 --remove