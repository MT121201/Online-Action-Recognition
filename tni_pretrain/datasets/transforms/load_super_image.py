import warnings
from typing import Optional

import mmengine.fileio as fileio
import numpy as np

import mmcv
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.builder import TRANSFORMS

from collections import deque
import torch
import cv2
import os
@TRANSFORMS.register_module()
class LoadSuperImage(BaseTransform):
    """Load an image from file.

    Required Keys:

    - img_path

    """

    def __init__(self,
                 super_image_grid = (3, 3),
                 ignore_empty: bool = False,
                 resize = (224,224),
                 num_group: int = None,
                 frame_per_group: int = None,
                 track_info_txt: str = None,):
        self.super_image_grid = super_image_grid
        self.sampling_dict = dict()
        self.track_id = None
        self.ignore_empty = ignore_empty
        self.num_group = num_group
        self.frame_per_group = frame_per_group
        self.track_info_txt = track_info_txt
        self.track_start_end = dict()
        self.fix_sampling_value = None
        self.loading_process = 0
        self.resize = resize
        # if exist num_group, frame_per_group, raise error only 1 of them is not None
        if self.num_group is not None:
            if self.frame_per_group is not None:
                raise ValueError('Only 1 of num_group and frame_per_group should be set')
        
    def add_sampling_dict(self, track_id, video_name, num_group, frame_per_group):
        #get start and end frame index
        with open(self.track_info_txt, 'r') as f:
            for line in f:
                line = line.strip().split(', ')
                if line[0] == video_name and int(line[1]) == int(track_id):
                    start_frame = int(line[2])
                    end_frame = int(line[3])
                    break
        self.track_start_end[video_name][track_id] = (start_frame, end_frame)

        # array of all track frames
        track_frames = np.linspace(start_frame, end_frame, end_frame - start_frame + 1, dtype=int)
        # slice track frames into groups
        if num_group is not None:
            # if num_group <9 => increase num_group = 9
            if num_group < 9:
                num_group = 9
                # print(f"num_group is too small, increase to {num_group}")

            groups = []
            frame_per_group = (self.track_start_end[track_id][1] - self.track_start_end[track_id][0]) // num_group
            self.fix_sampling_value = frame_per_group
            for i in range(0, len(track_frames), frame_per_group):
                groups.append(track_frames[i:i+frame_per_group])
            
            self.sampling_dict[video_name][track_id] = groups
        elif frame_per_group is not None:
            # if num_frames // frame_per_group < 9 => reduce frame_per_group until num_frames // frame_per_group = 9
            if len(track_frames) // frame_per_group < 9:
                frame_per_group = len(track_frames) // 9
                # print(f"frame_per_group is too large, reduce to {frame_per_group}")
            self.fix_sampling_value = frame_per_group
            groups = []
            for i in range(0, len(track_frames), frame_per_group):
                groups.append(track_frames[i:i+frame_per_group])
            
            self.sampling_dict[video_name][track_id] = groups
            
    def sampling(self, filename, track_id, video_name, file_path):
        # get group index contain ori frame
        group_index = (int(filename.split('_')[1]) - self.track_start_end[video_name][track_id][0] + 8) // self.fix_sampling_value 
        
        # get 8 group around group_index and include ori frame group
        groups = []

        if group_index < 4:
            groups = self.sampling_dict[video_name][track_id][:9]
        
        elif group_index > len(self.sampling_dict[video_name][track_id]) - 5:
            groups = self.sampling_dict[video_name][track_id][-9:]
            
        else:
            groups = self.sampling_dict[video_name][track_id][group_index-4:group_index+5]
            
        # random choose 1 frame from each group, and add ori frame to its position
        frames = []
        for i, group in enumerate(groups): 
            if i == group_index:
                frames.append(filename)
            else:
                file_name = track_id + '_' + str(np.random.choice(group)) + '_' + str('_'.join(filename.split('_')[2:]))
                file_check = os.path.join(file_path, file_name)

                if not os.path.exists(file_check):
                    frames.append(filename)
                else:
                    frames.append(file_name)
        # sort frames by frame index
        frames.sort(key=lambda x: int(x.split('_')[1]))
        # import pdb; pdb.set_trace()
        return frames
    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        import pdb; pdb.set_trace()
        filename = results['img_path']
        # get file path by delete last part of filename
        file_path = '/'.join(filename.split('/')[:-1])
        file_name = filename.split('/')[-1]
        #filename : trackid_frameix_class_..jpg
        # get track id
        track_id = filename.split('/')[-1].split('_')[0]
        # get video name
        video_name = '_'.join(filename.split('/')[-1].split('_')[3:5])
        # check if track id is in sampling dict, if not add it
        if video_name not in self.sampling_dict:
            self.sampling_dict[video_name] = {}
            self.track_start_end[video_name] = {}
        if track_id not in self.sampling_dict[video_name]:
            self.add_sampling_dict(track_id, video_name, self.num_group, self.frame_per_group)
        # import pdb; pdb.set_trace()
        # sampling 8 other frames from sampling dict
        frames_index = self.sampling(file_name, track_id, video_name, file_path)
        # load 9 frames 
        super_image = []
        for frame in frames_index:
            
            frame = os.path.join(file_path, frame)
            img = cv2.imread(frame)
            #resize
            img = cv2.resize(img, (224,224))
            super_image.append(img)
        # stack 9 frames into 1 image
        grid_image = np.vstack([np.hstack(row) for row in np.array_split(super_image , self.super_image_grid[0])])
        #resize to 224x224
        grid_image = cv2.resize(grid_image, self.resize)
        
        # add super image to results
        results['img'] = grid_image
        results['img_shape'] = grid_image.shape[:2]
        results['ori_shape'] = grid_image.shape[:2]
        # self.loading_process += 1
        # # if self.loading_process % 100 == 0:
        # #     print(f"loading process: {self.loading_process}")
        return results

