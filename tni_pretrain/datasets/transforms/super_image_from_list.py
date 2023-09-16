import warnings
from typing import Optional

import mmengine.fileio as fileio
import numpy as np

import mmcv
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.builder import TRANSFORMS

from collections import deque
import warnings
import torch
import cv2
import os
@TRANSFORMS.register_module()
class LoadSuperImageFromList(BaseTransform):
    def __init__(self,
                 croped_image_size = (100,128),
                 super_image_size = (224,224),
                 grid = (3, 3)):
        self.croped_image_size = croped_image_size
        self.super_image_size = super_image_size
        self.grid = grid
        if self.grid[0] != self.grid[1]:
            warnings.warn('grid[0] != grid[1], this may cause error in StackImage, if exist error check np.vstack')


    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filenames = results['img_path']
        ##############################################
        # this part is for convert to ONNX by mmdeploy
        if type(filenames) == str: #normaly when training or testing, filenames is a list
            filenames = [filenames]
        if len(filenames) != self.grid[0] * self.grid[1]:
            for i in range(self.grid[0] * self.grid[1] - len(filenames)):
                filenames.append(filenames[-1])
        ##############################################
        imgs = []
        for filename in filenames:
            img = cv2.imread(filename)
            if self.croped_image_size is not None:
                # print(img.shape)
                img = cv2.resize(img, self.croped_image_size)
            imgs.append(img)
        # transpose to grid
        grid_image = np.vstack([np.hstack(row) for row in np.array_split(imgs , self.grid[0])])
        #resize to 224x224
        grid_image = cv2.resize(grid_image, self.super_image_size)  
        results['img'] = grid_image
        results['img_shape'] = grid_image.shape[:2]
        results['ori_shape'] = grid_image.shape[:2]
        return results


