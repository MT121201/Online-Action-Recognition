import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


class Regcornize_action:
    def __init__(self,
                 model_path,
                 super_image_size,
                 threshold,
                 device,
                 small_image_size,
                 mean,
                 std):
        self.model_path = model_path
        self.super_image_size = super_image_size
        self.threshold = threshold
        self.small_image_size = small_image_size
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.index = 0
        self.track_id_queue_dict = {}
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

    def update_queue(self, img, track_id):
        # check is this new track id
        if track_id not in self.track_id_queue_dict:
            self.track_id_queue_dict[track_id] = []
        # update queue
        full_queue = False
        if len(self.track_id_queue_dict[track_id]) == self.super_image_size[0]*self.super_image_size[1]:
            self.track_id_queue_dict[track_id].pop(0)
            full_queue = True
        self.track_id_queue_dict[track_id].append(img)
        return full_queue

    def super_image(self, track_id):
        # take full queue array from dict
        queue_array = np.array(self.track_id_queue_dict[track_id])
        super_image = queue_array.reshape(self.super_image_size[0], self.super_image_size[1],
                                          self.small_image_size[0], self.small_image_size[1], 3).swapaxes(1, 2).reshape(3*224, 3*224, 3)
        return super_image

    def save_image(self, img, path):
        cv2.imwrite(path, img)

    def preprocess(self, img):
        # When reading image from cv2, it is BGR, convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        np_img = np.array(im_pil)
        # normalize image as training
        np_img = (np_img-self.mean)/self.std
        # convert to float32
        np_img = np_img.transpose(2, 0, 1).astype(np.float32)
        input_batch = np.expand_dims(np_img, 0)
        return input_batch

    def predict(self, img):
        # preprocess
        input_batch = self.preprocess(img)
        # run inference
        input_name = self.sess.get_inputs()[0].name
        output_name = self.sess.get_outputs()[0].name
        output = self.sess.run([output_name], {input_name: input_batch})[0]
        return output

    def run(self, img, result, track_id):
        # crop image
        crop_img = self.crop_detect_images(img, result)
        # resize image
        resize_img = self.resize_image(crop_img)
        # update queue
        full_queue = self.update_queue(resize_img, track_id)
        # # if queue is full
        if full_queue:
            # make super image
            super_image = self.super_image(track_id)
            # resize super image
            super_image = self.resize_image(super_image)
            # predict
            output = self.predict(super_image)
            # return if output score > threshold
            if np.max(output) > self.threshold:
                return np.argmax(output), np.max(output)
            else:
                return None, None

        else:
            return None, None
