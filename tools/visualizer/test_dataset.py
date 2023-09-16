from mmpretrain.registry import DATASETS
import random
import os
import mmcv
import click
import numpy as np
from tqdm import tqdm
from mmengine.config import Config
from mmpretrain.utils import register_all_modules
import cv2
# import cvut.draw as cvutdraw
register_all_modules()


@click.command()
@click.option("--config_file",
              default="configs/oad/repvgg_oad.py", help="your config file")
@click.option("--save_img_dir",
              default='./cache/debugdata1', help="save image dir")

def main(config_file, save_img_dir):
    cfg = Config.fromfile(config_file)
    classes = cfg['class_name']

    # build dataset
    dataset_cfg = cfg['train_dataloader']['dataset']
    if isinstance(dataset_cfg, dict):
        dataset = DATASETS.build(dataset_cfg)

    num_draw = 200
    
    random_selection = random.sample(range(len(dataset)), num_draw)

    print('len dataset: ', len(dataset))

    # draw all
    try:
        os.system(f'rm -rf {save_img_dir}')
    except:
        pass
    os.makedirs(save_img_dir, exist_ok=True)

    print('save images to: ', save_img_dir)

    for idx in tqdm(random_selection):

        sample = dataset[idx]

        # Read images
        img_name = str(idx) + '.jpg'
        img = sample['inputs'].data.cpu().numpy().transpose(1, 2, 0)
        label_idx = sample['data_samples'].gt_label.numpy().astype(np.int16)[0]
        label = classes[label_idx]
        # conver back
        img = cv2.putText(img.astype(np.uint8).copy(), str(label),
                          (10, 10), cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, (0, 255, 0), 2)

        save_file = os.path.join(save_img_dir, str(img_name))
        mmcv.imwrite(img, save_file)
        print(f'image is saved at: {save_file}')


if __name__ == '__main__':
    main()
