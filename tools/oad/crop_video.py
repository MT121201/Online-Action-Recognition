import cv2
import os
import argparse
import xml.etree.ElementTree as ET
import tqdm

class VideoProcessor:
    def __init__(self, video_path, xml_path, output_folder):
        self.video_path = video_path
        self.xml_path = xml_path
        self.output_folder = output_folder

        self.class_mapping = {}
        self.video_name = os.path.basename(video_path).split('.')[0]

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        if not os.path.exists('./cache'):
            os.makedirs('./cache')

    def convert_xml_to_txt(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        if root.findall('./meta/task/labels/label/name'):
            for idx, name in enumerate(root.findall('./meta/task/labels/label/name')):
                self.class_mapping[name.text] = idx
        else:
            for idx, name in enumerate(root.findall('./meta/job/labels/label/name')):
                self.class_mapping[name.text] = idx

        with open('./cache/helper_cropped.txt', 'w') as txt_file:
            for track in root.findall('./track'):
                track_id = track.get('id')
                class_label = track.get('label')
                class_id = self.class_mapping.get(class_label, -1)

                boxes = track.findall('./box')
                if not boxes:
                    continue

                start_frame = int(boxes[0].get('frame'))
                end_frame = int(boxes[-1].get('frame'))

                for box in boxes:
                    frame_idx = int(box.get('frame'))
                    xtl = float(box.get('xtl'))
                    ytl = float(box.get('ytl'))
                    xbr = float(box.get('xbr'))
                    ybr = float(box.get('ybr'))

                    line = f"{self.video_name}, {track_id}, {class_id}, {start_frame}, {end_frame}, {frame_idx}, {xtl}, {ytl}, {xbr}, {ybr}\n"
                    txt_file.write(line)

    def crop_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        annotations = open('./cache/helper_cropped.txt', 'r').readlines()

        print("Cropping frames...")
        pbar = tqdm.tqdm(total=len(annotations))

        for line in annotations:
            video_name, track_id, class_id, start_frame, _, frame_idx, xtl, ytl, xbr, ybr, *_ = line.strip().split(', ')
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()

            if ret:
                cropped_frame = frame[int(float(ytl)):int(float(ybr)), int(float(xtl)):int(float(xbr))].copy()
                output_filename = f"{track_id}_{frame_idx}_{class_id}_{video_name}_cropped.jpg"
                output_path = os.path.join(self.output_folder, output_filename)
                cv2.imwrite(output_path, cropped_frame)
            pbar.update(1)

        cap.release()
        pbar.close()

    def process_video(self):
        self.convert_xml_to_txt()
        self.crop_frames()
        print("Video cropped successfully, saved to: ", self.output_folder)

def main():
    parser = argparse.ArgumentParser(description='Crop frames from video')
    parser.add_argument('--video', type=str, help='Path to video')
    parser.add_argument('--xml', type=str, help='Path to annotations')
    parser.add_argument('--crop_save', type=str, help='Path to output folder')
    args = parser.parse_args()

    video_processor = VideoProcessor(args.video, args.xml, args.crop_save)
    video_processor.process_video()

if __name__ == '__main__':
    main()
