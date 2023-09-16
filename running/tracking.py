import argparse
import os

import cv2
import numpy as np
from loguru import logger

import onnxruntime

from yolox.data.data_augment import preproc as preprocess
from yolox.utils import mkdir, multiclass_nms, demo_postprocess, vis
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="/models/person/bytetrack_s.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--video_path",
        type=str,
        default='/data/its/oad/triet_test/video/self_record.mp4',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.1,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "-n",
        "--nms_thr",
        type=float,
        default=0.7,
        help="NMS threshould.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="608,1088",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

class args():
    def __init__(self,
                 track_thresh,
                 match_thresh, 
                 mot20, 
                 track_buffer
                 ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.mot20 = mot20
        self.track_buffer = track_buffer
        

class Tracking(object):
    def __init__(self, 
                 track_thresh = 0.5, 
                 match_thresh = 0.8,
                 track_buffer = 30,
                 min_box_area = 10,
                 mot20 = False,):
        self.min_box_area = min_box_area
        self.args = args(track_thresh, match_thresh, track_buffer, mot20)
        self.tracker = BYTETracker(self.args, frame_rate=30)
        self.frame_id = 0

    def _get_img_info(self, ori_img):
        img_info = {"id": 0}
        height, width = ori_img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = ori_img
        
        return img_info
    
    def tracking(self,  frame, results):
        img_info = self._get_img_info(frame)
        outputs = results
        if outputs is None:
            self.frame_id += 1
            return None
        online_targets = self.tracker.update(outputs, [img_info['height'], img_info['width']], [img_info['height'], img_info['width']])
        online_tlwhs = []
        online_ids = []
        online_scores = []
        results = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
        self.frame_id += 1
        results.append((self.frame_id, online_tlwhs, online_ids, online_scores))
        return results


def imageflow(predictor, args):
    cap = cv2.VideoCapture(args.video_path)
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    # fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = args.output_dir
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, args.video_path.split("/")[-1])
    logger.info(f"video save_path is {save_path}")
    # vid_writer = cv2.VideoWriter(
    #     save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    # )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            timer.tic()
            online_tlwhs, online_ids, online_scores = predictor.tracking(frame)
            print("online_tlwhs", online_tlwhs)
            print("online_ids", online_ids)
            print("online_scores", online_scores)
            # results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
            timer.toc()
            # online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,
            #                           fps=1. / timer.average_time)
            # vid_writer.write(online_im)
        else:
            break
        frame_id += 1


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    predictor = Tracking(model=args.model, score_thr=args.score_thr, nms_thr=args.nms_thr, input_shape=args.input_shape, track_thresh=args.track_thresh, match_thresh=args.match_thresh) 
    imageflow(predictor, args)