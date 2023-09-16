import cv2
import numpy as np

class CenterCrop:
    def __init__(self, crop_size, offset = (0, 0)):
        self.crop_size = crop_size
        self.special_crop_size = True
        self.with_offset = False
        self.offset = offset
        if self.offset != (0, 0):
            self.with_offset = True
        # if crop size input is not a tuple with 2 integers, that is special input
        if isinstance(self.crop_size, tuple):
            if isinstance(self.crop_size[0], int) and isinstance(self.crop_size[1], int):
                if self.crop_size[0] == 0 or self.crop_size[1] == 0:
                    self.special_crop_size = True
                else:
                    self.special_crop_size = False

    def convert_crop_size(self, frame):
        """
        Special crop size input:
        1. 'tuple'
        1.1 (int, 0) or (0, int): crop size is one dimension and keep aspect ratio
        1.2 (,int) or (int, ): Please Using 0 instead of None
        1.3 (-1, int) or (int, -1): crop size is one dimension and full size of other dimension
        1.4 (int, float) or (float, int): one size of int and one size take ratio of float
        1.5 (float, float): take ratio of both dimension
        2. 'list'
        Raise Please Using Tuple or Integer or Float
        3. 'int'
        Crop size is square size
        4. 'float'
        Crop size is ratio of original image size
        5. 'str'
        Raise Please Using Tuple or Integer or Float

        """
        crop_size = self.crop_size
        frame_height, frame_width = frame.shape[:2]
        if isinstance(crop_size, tuple):
            if len (crop_size) != 2:
                raise ValueError("Crop size must be 2 dimension, please use 0 instead of None")
            H, W = crop_size
            # If float take ratio of frame size
            if isinstance(H, float):
                H = int(frame_height * crop_size[0])
            if isinstance(W, float):
                W = int(frame_width * crop_size[1])
            # If one size is -1, take full size of that dimension
            if H == -1:
                H = frame_height
            elif W == -1:
                W = frame_width
            elif H == -1 and W == -1:
                raise ValueError("Only one size can be -1")
            # If one size is 0, keep aspect ratio
            if H == 0:
                H = int(frame_height * W / frame_width)
            elif W == 0:
                W = int(frame_width * H / frame_height)
            elif H == 0 and W == 0:
                raise ValueError("Only one size can be 0")
            crop_size = (H, W)

        elif isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        
        elif isinstance(crop_size, float):
            crop_size = (int(frame_height * crop_size), int(frame_width * crop_size))
        
        else:
            raise ValueError("Please Using Tuple or Integer or Float")
        
        self.__check_size__(crop_size, frame)

    def __check_size__(self, crop_size, frame):
        frame_height, frame_width = frame.shape[:2]
        if crop_size[0] > frame_height or crop_size[1] > frame_width:
            raise ValueError("Crop size is larger than frame size")
        if isinstance(crop_size[0], int) and isinstance(crop_size[1], int):
            # From now, dont need to convert crop size anymore
            self.special_crop_size = False
            self.crop_size = crop_size
        else:
            raise ValueError("Wrong type of crop size expected can convert to int,int but got", type(crop_size[0]), type(crop_size[1]))

    def __crop__(self, frame):
        if self.special_crop_size:
            self.convert_crop_size(frame)
        frame_height, frame_width = frame.shape[:2]
        crop_height, crop_width = self.crop_size
        if self.with_offset:
            center_x = int(frame_width // 2 + self.offset[0])
            center_y = int(frame_height // 2 + self.offset[1])

            x1 = max(center_x - crop_width // 2, 0)
            y1 = max(center_y - crop_height // 2, 0)
            x2 = min(center_x + crop_width // 2, frame_width)
            y2 = min(center_y + crop_height // 2, frame_height)
        else:
            x1 = int((frame_width - crop_width) / 2)
            y1 = int((frame_height - crop_height) / 2)
            x2 = x1 + crop_width
            y2 = y1 + crop_height
        return np.array([x1, y1, x2, y2])

def main():
    test_image = cv2.imread('/data/its/oad/triet_test/cropped_image/0_17_2_self_record2.mp4_cropped.jpg')
    center_crop = CenterCrop(0.1)
    print(test_image.shape)
    print(center_crop.__crop__(test_image))

if __name__ == "__main__":
    main()
                    

