from action_regcornize import Regcornize_action
from detection import Detection_model

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
def draw_bbox(img, bbox, predict, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox[:4]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(img, str(predict), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)
    return img
def arg_parse():
    parser = argparse.ArgumentParser(description='Run pipeline')
    parser.add_argument('--detection', type=str, default="/models/person/yolox_s_hdet_vga_60e_onnxonly_1280x800.onnx", help='path to detection model')
    parser.add_argument('--action', type=str, default="/models/person/action_regcornize.onnx", help='path to action model')
    parser.add_argument('--s_size', type=tuple, default=(3,3), help='size of super image')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold of action model')
    parser.add_argument('--input', type=str, help='path to input video')
    parser.add_argument('--save', type=str, default='./cache/video', help='path to save video')
    parser.add_argument('--class_name', type=str, default=None, help='class name txt file')
    args = parser.parse_args()
    return args
def main():
    args = arg_parse()
    detection_onnx = args.detection
    action_onnx = args.action
    super_image_size = args.s_size
    threshold = args.threshold
    input_video = args.input
    save_path = args.save
    class_name = args.class_name
    # check
    if not os.path.exists(detection_onnx):
        raise ValueError("Detection model not found!")
    if not os.path.exists(action_onnx):
        raise ValueError("Action model not found!")
    if not os.path.exists(input_video):
        raise ValueError("Input video not found!")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if class_name is None:
        raise ValueError("Class name not found!")
    # init model
    detection_model = Detection_model(model_path=detection_onnx, H=800, W=1280)
    regcornize_action = Regcornize_action(model_path=action_onnx, super_image_size=super_image_size, threshold=threshold)

    with open(class_name, 'r') as f:
        class_name = f.read().splitlines()
    
#   read and write video
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_file = os.path.join(save_path, 'output.mp4')
    out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
    #pbar
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # detection
            results = detection_model.detect(frame)
            for result in results:
                # regcornize
                predict = regcornize_action.run(frame, result)
                # # draw bbox
                if predict is not None:
                    frame = draw_bbox(frame, result, class_name[predict])
            out.write(frame)
            pbar.update(1)
        else:
            break
    
    cap.release()
    out.release()
    pbar.close()
if __name__ == "__main__":
    main()
        
