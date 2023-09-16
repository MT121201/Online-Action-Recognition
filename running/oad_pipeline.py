from action_regcornize import Regcornize_action
from detection import Detection_model
from tracking import Tracking
from center_crop import CenterCrop
import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import importlib.util
from loguru import logger
from yolox_timer import Timer


def arg_parse():
    parser = argparse.ArgumentParser(description='Run pipeline')
    parser.add_argument('--configs',
                        type=str, default="configs/infer_oad/repvva0_oad_with_persondet_infer.py")
    parser.add_argument('--input',
                        type=str, help='path to input video')
    parser.add_argument('--save',
                        type=str, default='./cache/video', help='path to save video')
    parser.add_argument('--use_detector', action='store_true',
                        help='use detector for tracking')
    args = parser.parse_args()
    return args


def draw_bbox(frame, infos, class_name):
    """
    Draw bounding boxes on image, with track id, action label, and score
    """
    for track_info in infos:
        track_id, bbox, predict, score = track_info
        label = class_name[predict]
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"ID: {track_id} | Label: {label} | Score: {score:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


def process_video_with_detector(input_video, save_path, tracker,
                                detection_model, regcornize_action, class_name):
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_name = input_video.split('/')[-1].replace('.mp4', '_output.mp4')
    out_file = os.path.join(save_path, out_name)
    out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_idx = 0
    timer = Timer()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame_idx % 100 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(
                    frame_idx, 1. / max(1e-5, timer.average_time)))
            timer.tic()
            # detection
            detect_results = detection_model.detect(frame)
            # tracking
            tracking_result = tracker.tracking(frame, detect_results)
            # avoid error when no detection
            plot_infos = []
            if tracking_result is not None:
                for i, track_id in enumerate(tracking_result[0][2]):
                    tlwh = tracking_result[0][1][i]
                    bbox = tlwh[0], tlwh[1], tlwh[0]+tlwh[2], tlwh[1]+tlwh[3]
                    # action recognition
                    predict, score = regcornize_action.run(
                        frame, bbox, track_id)
                    # draw bbox
                    if predict is not None:
                        plot_infos.append((track_id, bbox, predict, score))
            draw_bbox(frame, plot_infos, class_name)
            timer.toc()
            out.write(frame)
            pbar.update(1)
            frame_idx += 1
        else:
            break

    cap.release()
    out.release()
    pbar.close()
    logger.info('Done processing {} frames in {:.2f} seconds at {:.2f} fps'.format(
        frame_idx, timer.total_time, frame_idx / timer.total_time))
    logger.info('Results saved to {}'.format(out_file))


def process_video_without_detector(input_video, save_path, centercrop, regcornize_action, class_name):
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_name = input_video.split('/')[-1].replace('.mp4', '_output.mp4')
    out_file = os.path.join(save_path, out_name)
    out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_idx = 0
    timer = Timer()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame_idx % 100 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(
                    frame_idx, 1. / max(1e-5, timer.average_time)))
            timer.tic()
            # center crop
            bbox = centercrop.__crop__(frame)
            # action recognition
            predict, score = regcornize_action.run(frame, bbox, 0)
            # draw bbox
            if predict is not None:
                plot_infos = [(0, bbox, predict, score)]
                draw_bbox(frame, plot_infos, class_name)
            timer.toc()
            out.write(frame)
            pbar.update(1)
            frame_idx += 1
        else:
            break
    cap.release()
    out.release()
    pbar.close()
    logger.info('Done processing {} frames in {:.2f} seconds at {:.2f} fps'.format(
        frame_idx, timer.total_time, frame_idx / timer.total_time))
    logger.info('Results saved to {}'.format(out_file))


def main():
    args = arg_parse()
    # TODO: Read config file and parse detection, action models, etc.
    config_file = args.configs
    input_video = args.input
    save_path = args.save

    # get config
    config_spec = importlib.util.spec_from_file_location('config', config_file)
    config_module = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)

    # check
    if args.use_detector:
        if not os.path.exists(config_module.detector['model']):
            raise ValueError("Detector model not found!")
    if not os.path.exists(config_module.action['model']):
        raise ValueError("Action model not found!")
    if not os.path.exists(input_video):
        raise ValueError("Input video not found!")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if config_module.class_name is None:
        raise ValueError("Class name not found!")

    # init model
    if args.use_detector:
        detection_model = Detection_model(model_path=config_module.detector['model'],
                                          H=config_module.detector['size'][0],
                                          W=config_module.detector['size'][1],
                                          device=config_module.device
                                          )

        tracking_model = Tracking(track_thresh=config_module.tracking['track_thresh'],
                                  match_thresh=config_module.tracking['match_thresh'],
                                  mot20=False,
                                  track_buffer=config_module.tracking['track_buffer'],
                                  min_box_area=config_module.tracking['min_box_area'],
                                  )
    else:
        centercrop = CenterCrop(crop_size=config_module.centercrop['crop_size'],
                                offset=config_module.centercrop['offset'])

    regcornize_action = Regcornize_action(model_path=config_module.action['model'],
                                          super_image_size=config_module.action['super_image_size'],
                                          small_image_size=config_module.action['small_image_size'],
                                          threshold=config_module.action['threshold'],
                                          device=config_module.device,
                                          mean=config_module.action['mean'],
                                          std=config_module.action['std']
                                          )

    if args.use_detector:
        process_video_with_detector(input_video, save_path, tracking_model, detection_model,
                                    regcornize_action, config_module.class_name)
    else:
        process_video_without_detector(
            input_video, save_path, centercrop, regcornize_action, config_module.class_name)


if __name__ == "__main__":
    main()
