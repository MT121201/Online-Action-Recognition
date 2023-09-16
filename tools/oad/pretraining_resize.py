import cv2
import numpy as np
import argparse
import os
import tqdm

class ImageProcessor:
    def __init__(self, input_folder, save_folder, H, W, remove=False):
        self.input_folder = input_folder
        self.save_folder = save_folder
        self.H = H
        self.W = W
        self.remove = remove

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def load_image(self, image_path):
        img = cv2.imread(image_path)
        # If H or W is None, then resize and keep aspect ratio
        if self.H is None or self.W is None:
            h, w, _ = img.shape
            if self.H is None:
                self.H = int(h * self.W / w)
            else:
                self.W = int(w * self.H / h)
        img = cv2.resize(img, (self.W, self.H))
        return img
    
    def processing_image(self):
        print('Resize images from folder:', self.input_folder)
        load_bar = tqdm.tqdm(total=len(os.listdir(self.input_folder)))
        for filename in os.listdir(self.input_folder):
            img = self.load_image(os.path.join(self.input_folder, filename))
            if img is not None:
                cv2.imwrite(os.path.join(self.save_folder, filename), img)
                load_bar.update(1)
        load_bar.close()
        print('Successfully resized images and saved to folder:', self.save_folder)
        if self.remove:
            print('Removing folder:', self.input_folder)
            os.system('rm -rf ' + self.input_folder)

def argphase():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-input', type=str, default='data/imagenet', help='path to the folder containing images')
    parser.add_argument('-save', type=str, default=None, help='path to the folder containing images')
    parser.add_argument('-H', type=int, default=None, help='H to resize')
    parser.add_argument('-W', type=int, default=None, help='W to resize')
    parser.add_argument('--remove', action='store_true', help='Remove cropped images folder after have resized cropped images')
    args = parser.parse_args()
    return args

def main():
    args = argphase()
    input_folder = args.input
    save_folder = args.save

    if input_folder is None:
        raise ValueError('input folder cannot be None')

    if save_folder is None:
        # Save in the same root folder named + '_resized'
        save_folder = input_folder + '_resized'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if args.H is None and args.W is None:
        raise ValueError('H and W cannot be None at the same time')

    image_processor = ImageProcessor(input_folder, save_folder, args.H, args.W, args.remove)
    image_processor.processing_image()

if __name__ == '__main__':
    main()
