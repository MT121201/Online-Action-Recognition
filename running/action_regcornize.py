import cv2
import numpy as np
from collections import deque
import onnxruntime as ort
import torch
from PIL import Image
import os
import time
class Regcornize_action:
    def __init__ (self, model_path, super_image_size, threshold, device):
        self.model_path = model_path
        self.super_image_size = super_image_size
        self.threshold = threshold
        self.queue = deque(maxlen=super_image_size[0]*super_image_size[1])
        # Build model with the specified device (CPU or GPU)
        if device == 'cpu':
            providers = ['CPUExecutionProvider']
        elif device == 'gpu':
            providers = ['CUDAExecutionProvider']
        else:
            raise ValueError("Invalid device choice. Use 'cpu' or 'gpu'.")

        self.sess = ort.InferenceSession(self.model_path, providers=providers)
    
    def crop_detect_images(self, img, result):
        # crop image
        x1, y1, x2, y2 = result[:4]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(img.shape[1], int(x2))
        y2 = min(img.shape[0], int(y2))
        crop_img = img[y1:y2, x1:x2]
        return crop_img
    
    def resize_image(self, img, height=224, width=224):
        # resize and expand batch dimension
        img = cv2.resize(img, (width, height))
        return img
    
    def update_queue(self, img):
        # update queue
        if len(self.queue) < self.super_image_size[0]*self.super_image_size[1]:
            self.queue.append(img)
            return False
        else:
            self.queue.popleft()
            self.queue.append(img)
            return True
    def super_image(self):
        # make super image 3x3 from queue
        queue_array = np.array(self.queue)
        super_image = queue_array.reshape(self.super_image_size[0], self.super_image_size[1], 224, 224, 3).swapaxes(1, 2).reshape(3*224, 3*224, 3)
        return super_image
    
    def save_image(self, img, path):
        cv2.imwrite(path, img)

    def predict(self, img):
        # preprocess
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        np_img = np.array(im_pil)
        np_img = np_img.transpose([2, 0, 1]).astype('float32')
        input_batch = np.expand_dims(np_img, 0)
        # run inference
        input_name = self.sess.get_inputs()[0].name
        output_name = self.sess.get_outputs()[0].name
        output = self.sess.run([output_name], {input_name: input_batch})[0]
        
        return output
    
    def run(self, img, result):
        # crop image
        crop_img = self.crop_detect_images(img, result)
        # resize image
        resize_img = self.resize_image(crop_img)
        # update queue
        full_queue = self.update_queue(resize_img)
        # # if queue is full
        if full_queue:
            # make super image
            super_image = self.super_image()
            # predict
            output = self.predict(super_image)
            # return if output score > threshold
            
            if np.max(output) > self.threshold:
                # image_name = str(np.argmax(output)) +'_' + str(time.time()) + '.jpg'
                # save_path = os.path.join('./cache/infer', image_name)
                # self.save_image(super_image, save_path)
                return np.argmax(output)
            else:
                return None
            
        else:
            return None
