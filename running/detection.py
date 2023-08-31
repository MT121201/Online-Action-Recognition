# This is detection model, will recieve video input and output bounding box of each frame.
import os
import cv2
# import cvut.draw as cvutdraw
import numpy as np
import onnxruntime as ort
from PIL import Image
import time
class Detection_model:
    def __init__(self, model_path, H, W, save_path):
        self.model_path = model_path
        self.H = H
        self.W = W
        self.save_path = None

        # build model
        providers = [
            'CUDAExecutionProvider'
        ]
        self.predictor = ort.InferenceSession(self.model_path, providers=providers)
        self.io_binding = self.predictor.io_binding()
        self.input_tensor = self.predictor.get_inputs()[0]
    
    def preprocess(self, pil_im, height=384, width=128):
        # resize and expand batch dimension
        rgb_im = pil_im.convert('RGB')
        np_img = np.array(rgb_im)
        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(np_img, (width, height))
        img = np.expand_dims(img, 0).transpose([0, 3, 1, 2]).astype('float32')
        return img
    
    def post_process(self, results, score_thr=0.5, scale_x=None, scale_y=None):
        results = results.reshape(-1, 5)
        results = results[results[:, -1] >= score_thr]
        results[:, 0] *= scale_x
        results[:, 2] *= scale_x
        results[:, 1] *= scale_y
        results[:, 3] *= scale_y
        return results

    def detect(self, request_img):
        h, w, c = request_img.shape

        img = cv2.cvtColor(request_img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)

        # preprocessing
        np_img = self.preprocess(im_pil, self.H, self.W)

        # run inference
        self.io_binding.bind_cpu_input('input', np_img)
        for output in ['dets', 'labels']:
            self.io_binding.bind_output(output)
        self.predictor.run_with_iobinding(self.io_binding)
        results = self.io_binding.copy_outputs_to_cpu()[0]

        # post-processing
        scale_x = w/self.W
        scale_y = h/self.H
        results = self.post_process(results, 0.5, scale_x, scale_y)
        return results
