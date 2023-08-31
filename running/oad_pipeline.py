from action_regcornize import Regcornize_action
from detection import Detection_model

import cv2
import numpy as np

from tqdm import tqdm
def draw_bbox(img, bbox, predict, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = bbox[:4]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # import pdb; pdb.set_trace()
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(img, str(predict), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2)
    # save_path = './cache/frame.jpg'
    # cv2.imwrite(save_path, img)
    return img

def main():
    detection_model = Detection_model(model_path="/models/person/yolox_s_hdet_vga_60e_onnxonly_1280x800.onnx", H=800, W=1280, save_path=None)
    regcornize_action = Regcornize_action(model_path="/home/tni/Workspace/triet/Vehicle-Classification/onnx_model/action_regcornize.onnx", super_image_size=(3,3), threshold=0.5)
    input_video = "/data/its/oad/triet_test/video/call_test.mp4"
    save_path = "./cache/video/call_test.mp4"
    class_name = ("call", "read", "purcharse", "read")
#   read and write video
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
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
        
