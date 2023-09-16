import os
import cv2
import cvut.draw as cvutdraw
import numpy as np
import onnxruntime as ort
from PIL import Image

model_path = "/models/person/yolox_s_hdet_vga_60e_onnxonly_1280x800.onnx"

# input from config
IMG_W, IMG_H = 1280, 800

save_img_path = './cache/test_img.jpg'

def preprocess(pil_im, height=384, width=128):
    # resize and expand batch dimension
    rgb_im = pil_im.convert('RGB')
    np_img = np.array(rgb_im)
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(np_img, (width, height))
    img = np.expand_dims(img, 0).transpose([0, 3, 1, 2]).astype('float32')
    return img


def post_process(results, score_thr=0.5, scale_x=None, scale_y=None):
    results = results.reshape(-1, 5)
    results = results[results[:, -1] >= score_thr]
    results[:, 0] *= scale_x
    results[:, 2] *= scale_x
    results[:, 1] *= scale_y
    results[:, 3] *= scale_y
    return results

# build model
providers = [
    'CUDAExecutionProvider'
]
predictor = ort.InferenceSession(model_path, providers=providers)
io_binding = predictor.io_binding()
input_tensor = predictor.get_inputs()[0]

# read image
request_img = cv2.imread("img/person.jpg")
h, w, c = request_img.shape

img = cv2.cvtColor(request_img, cv2.COLOR_BGR2RGB)
im_pil = Image.fromarray(img)

# preprocessing
np_img = preprocess(im_pil, IMG_H, IMG_W)

# run inference
io_binding.bind_cpu_input('input', np_img)
for output in ['dets', 'labels']:
    io_binding.bind_output(output)
predictor.run_with_iobinding(io_binding)
results = io_binding.copy_outputs_to_cpu()[0]

# post-processing
scale_x = w/IMG_W
scale_y = h/IMG_H
results = post_process(results, 0.5, scale_x, scale_y)

# draw result
result_img = cvutdraw.draw_bboxes(request_img, results[:, :4])
os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
cv2.imwrite(save_img_path, result_img)
print('your image is saved under: ', save_img_path)
