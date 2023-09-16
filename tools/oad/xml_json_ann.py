import cv2
import numpy as np
import argparse
import os
import tqdm
import random
import xml.etree.ElementTree as ET
import json

class DataConverter:
    def __init__(self, xml_path, video_path, split = False , save_path='./cache/json/'):
        self.xml_path = xml_path
        self.video_path = video_path
        self.save_path = save_path
        self.split = split

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def convert_xml_json(self):
        video_name = self.video_path.split('/')[-1]
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        class_mapping = {}
        class_label = []

        if root.findall('./meta/task/labels/label/name'):
            for idx, name in enumerate(root.findall('./meta/task/labels/label/name')):
                class_mapping[name.text] = idx
                class_label.append(name.text)
        else:
            for idx, name in enumerate(root.findall('./meta/job/labels/label/name')):
                class_mapping[name.text] = idx
                class_label.append(name.text)

        data = {
            "class_mapping": class_mapping,
            "track_data": {}
        }

        for track in root.findall('./track'):
            file_paths = []
            labels = []
            track_id = track.get('id')
            label = track.get('label')
            class_id = class_label.index(label)
            boxes = track.findall('./box')
            if not boxes:
                continue
            for box in boxes:
                frame_idx = int(box.get('frame'))
                file_path = f"{track_id}_{frame_idx}_{class_id}_{video_name}_cropped.jpg"
                file_paths.append(file_path)
                labels.append(label)

            if track_id not in data["track_data"]:
                data["track_data"][track_id] = {
                    "image_paths": [],
                    "labels": [],
                }

            data["track_data"][track_id]["image_paths"].append(file_paths)
            data["track_data"][track_id]["labels"].append(labels)

        return data

    @staticmethod
    def split_train_test(data, train_ratio=0.8):
        class_mapping = data["class_mapping"]
        train_data = {
            "class_mapping": class_mapping,
            "track_data": {}
        }
        test_data = {
            "class_mapping": class_mapping,
            "track_data": {}
        }
        for track_id, track_data in data["track_data"].items():
            image_paths = track_data["image_paths"][0]
            labels = track_data["labels"][0]
            length = len(image_paths)
            num_of_test = int(length * (1 - train_ratio))
            test_idx = random.sample(range(length), num_of_test)
            train_idx = [i for i in range(length) if i not in test_idx]
            if track_id not in train_data["track_data"]:
                train_data["track_data"][track_id] = {
                    "image_paths": [],
                    "labels": [],
                }
            if track_id not in test_data["track_data"]:
                test_data["track_data"][track_id] = {
                    "image_paths": [],
                    "labels": [],
                }
            for idx in train_idx:
                train_data["track_data"][track_id]["image_paths"].append(image_paths[idx])
                train_data["track_data"][track_id]["labels"].append(labels[idx])
            for idx in test_idx:
                test_data["track_data"][track_id]["image_paths"].append(image_paths[idx])
                test_data["track_data"][track_id]["labels"].append(labels[idx])

        return train_data, test_data

    def save_json(self, data, split=False):
        if split:
            train_data, test_data = self.split_train_test(data)
            train_json = os.path.join(self.save_path, "train.json")
            with open(train_json, 'w') as w_json:
                json.dump(train_data, w_json, indent=4)
            print("Train file saved at: ", train_json)
            test_json = os.path.join(self.save_path, "test.json")
            with open(test_json, 'w') as w_json:
                json.dump(test_data, w_json, indent=4)
            print("Test file saved at: ", test_json)
        else:
            json_file = os.path.join(self.save_path, "data.json")
            with open(json_file, 'w') as w_json:
                json.dump(data, w_json, indent=4)
            print("Json file saved at: ", json_file)
    
    def make_json_ann(self):
        data = self.convert_xml_json()
        self.save_json(data, split=self.split)
        
def parse_args():
    parser = argparse.ArgumentParser(description='Convert xml to json')
    parser.add_argument('--xml', help='path to xml file')
    parser.add_argument('--save', default='./cache/json/', help='path to save json file')
    parser.add_argument('--video', help='path to video file')
    parser.add_argument('--split', action='store_true', help="Specify if the data is split")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    xml_file = args.xml
    video_file = args.video
    save_path = args.save

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(xml_file):
        raise Exception("xml file not found")

    if not os.path.exists(video_file):
        raise Exception("video file not found")

    data_converter = DataConverter(xml_file, video_file, save_path)
    data = data_converter.convert_xml_json()

    if args.split:
        data_converter.save_json(data, split=True)
    else:
        data_converter.save_json(data, split=False)

if __name__ == "__main__":
    main()
