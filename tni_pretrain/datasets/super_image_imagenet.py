import json
import numpy as np
from typing import Optional
from typing import Optional, Sequence, Union

from mmpretrain.registry import DATASETS
from mmpretrain.datasets.base_dataset import BaseDataset
from mmengine.fileio import get_file_backend


@DATASETS.register_module()
class SuperImage(BaseDataset):
    """Load an image from file and return super image grid."""

    def __init__(self,
                 data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 ann_file: str = '',
                 with_label=True,
                 extensions: Sequence[str] = ('.jpg', '.jpeg', '.png', '.ppm',
                                              '.bmp', '.pgm', '.tif'),
                 metainfo: Optional[dict] = None,
                 lazy_init: bool = False,
                 super_image_grid=(3, 3),
                 frame_per_group: int = 16,
                 **kwargs):

        self.super_image_grid = super_image_grid
        self.frame_per_group = frame_per_group

        self.extensions = tuple(set([i.lower() for i in extensions]))
        self.with_label = with_label

        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            metainfo=metainfo,
            # Force to lazy_init for some modification before loading data.
            lazy_init=True,
            **kwargs)

        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()

    def ckeck_ann_file(self, data):
        # check format
        if not isinstance(data, dict):
            raise ValueError('Annotation file should be a json file')
        # Json file structure should be:
        # {class_mapping:{},
        # track_data: {track_id: {image_paths: [path1, path2, ...], labels: [label1, label2, ...]}}}
        # Check existence of class_mapping
        if 'class_mapping' not in data:
            raise ValueError('Annotation file should contain class_mapping')
        # Get all track_ids
        track_ids = list(data['track_data'].keys())
        # Check all info of each track_id
        for track_id in track_ids:
            print("Checking track_id: ", track_id)
            if 'image_paths' not in data['track_data'][track_id]:
                raise ValueError('Annotation file should contain image_paths')
            # check labels
            if 'labels' not in data['track_data'][track_id]:
                raise ValueError('Annotation file should contain labels')
            # check length
            if len(data['track_data'][track_id]['image_paths']) != len(data['track_data'][track_id]['labels']):
                raise ValueError(
                    'The number of image_paths and labels should be the same')
            print("OK")

    def _load_json_annotations(self):
        """Read the json annotation file."""
        if not self.ann_file:
            raise ValueError('Please specify ann_file')
        if not self.with_label:
            raise ValueError('Please set with_label=True')
        # raise error if anntation file is not json
        if not self.ann_file.endswith('.json'):
            raise ValueError('Please set ann_file as a json file')
        # load annotations
        with open(self.ann_file, 'r') as f:
            data = json.load(f)
        # check format
        self.ckeck_ann_file(data)
        # Get class names list with corresponding class index.
        class_list = self.mappping_label(data)
        # Load samples as this format:
        # Dict[track_id][image_paths, labels]
        samples_dict = {}
        for track_id, info in data['track_data'].items():
            track_id = int(track_id)
            samples_dict[track_id] = {}
            samples_dict[track_id]['image_paths'] = info['image_paths']
            samples_dict[track_id]['labels'] = info['labels']
        return samples_dict, class_list

    def mappping_label(self, data):
        """Get class names list with corresponding class index."""
        class_mapping = data['class_mapping']
        class_names = []
        for class_name, class_idx in class_mapping.items():
            class_names.append(class_name)
        return class_names

    def load_data_list(self):
        """Load image paths and gt_labels."""
        if not self.ann_file:
            raise ValueError('Please specify ann_file')
        elif self.with_label:
            samples_dict, class_list = self._load_json_annotations()
        else:
            raise ValueError('Please set with_label=True')
        # Pre-build file backend to prevent verbose file backend inference.
        backend = get_file_backend(self.img_prefix, enable_singleton=True)
        data_list = []
        ##############################################
        # this part is for sampling super image
        # do for each track_id
        for track_id, samples in samples_dict.items():
            # sampling
            len_ds = len(samples['image_paths'])
            # num samples is the number of super images
            num_per_grid = self.super_image_grid[0] * self.super_image_grid[1]
            # if len_ds <num_per_grid repeat last frame to get total num_per_grid images
            if len_ds < num_per_grid:
                for i in range(num_per_grid - len_ds):
                    samples['image_paths'].append(samples['image_paths'][-1])
                    samples['labels'].append(samples['labels'][-1])
                len_ds = len(samples['image_paths'])

            num_group = len_ds // self.frame_per_group
            # if num_group < num_per_grid: reduce frame_per_group to get num_per_grid groups
            if num_group < num_per_grid:
                self.frame_per_group = len_ds // num_per_grid
                num_group = len_ds // self.frame_per_group

            for i in range(num_group):
                sampling_samples = []
                # chosing num_per_grid frames from each group around i
                if i < num_per_grid // 2:
                    j = np.linspace(0, num_per_grid - 1,
                                    num_per_grid,  dtype=int)
                elif i > num_group - num_per_grid // 2 - 1:
                    j = np.linspace(num_group - num_per_grid,
                                    num_group - 1, num_per_grid,  dtype=int)
                else:
                    j = np.linspace(i - num_per_grid // 2, i +
                                    num_per_grid // 2, num_per_grid,  dtype=int)
                for k in j:
                    random_index = np.random.randint(
                        k * self.frame_per_group, (k + 1) * self.frame_per_group)
                    sampling_samples.append(
                        (samples['image_paths'][random_index], samples['labels'][random_index]))
                img_paths = []
                gt_labels = []
                for sample in sampling_samples:
                    filename, gt_label = sample
                    img_path = backend.join_path(self.img_prefix, filename)
                    img_paths.append(img_path)
                    gt_labels.append(class_list.index(gt_label))
                # the highest frequency label is the label of the super image
                # TODO: check can we use this method to get super image label, or take label of last frame
                gt_label = max(set(gt_labels), key=gt_labels.count)
                info = {'img_path': img_paths, 'gt_label': gt_label}
                data_list.append(info)
        return data_list

    def is_valid_file(self, filename: str) -> bool:
        """Check if a file is a valid sample."""
        return filename.lower().endswith(self.extensions)
