import json
import os
import random

from torch.utils.data import Dataset

from howtocaption.data_loader.video_datasets.utils import get_video_frames
from howtocaption.data_loader.transforms import init_transform_dict


class YouCook2(Dataset):
    def __init__(self, data_root, split, num_frames=4, transforms=None,
                 dataset_name='YouCook2',
                 output_for_visualization=False,
                 max_text_length=-1
                 ):
        super(YouCook2, self).__init__()

        with open(os.path.join(data_root, 'youcookii_annotations_trainval.json'), 'rb') as fin:
            all_data = json.load(fin)['database']

        with open(os.path.join(data_root, 'splits', f'{split}_list.txt'), 'r') as fin:
            video_ids = fin.readlines()
            video_ids = [os.path.basename(name.strip()) for name in video_ids if name.strip() != '']

            self.data = []
            for video_id in video_ids:
                for clip_data in all_data[video_id]['annotations']:
                    clip_id = f'{video_id}_{clip_data["id"]}'
                    clip_data['video_id'] = video_id
                    clip_data['clip_id'] = clip_id
                    clip_data['recipe_type'] = all_data[video_id]['recipe_type']
                    self.data.append(clip_data)

        self.data_root = data_root
        self.split = split
        self.transforms = transforms
        self.num_frames = num_frames
        self.dataset_name = dataset_name
        self.output_for_visualization = output_for_visualization
        self.max_text_length = max_text_length
        if output_for_visualization:
            self.vis_transform = init_transform_dict()['visualization']

    def _get_video_path(self, sample):
        if self.split == 'train':
            folder = 'training'
        elif self.split == 'val':
            folder = 'validation'
        elif self.split == 'test':
            folder = 'testing'
        else:
            raise NotImplementedError
        rel_path = os.path.join(folder, sample['recipe_type'], sample['video_id'], sample['video_id'])
        if os.path.exists(os.path.join(self.data_root, rel_path + '.mp4')):
            rel_path = rel_path + '.mp4'
        else:
            rel_path = rel_path + '.mkv'

        return os.path.join(self.data_root, rel_path), rel_path

    def _get_caption(self, sample):
        return sample['sentence']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = idx % len(self.data)

        sample = self.data[idx]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)

        start_clip, end_clip = sample['segment']
        num_sec = end_clip - start_clip - 0.1 # just in case
        if self.split == 'train':
            # sample start
            fps = (self.num_frames + 1) / num_sec
            start_clip = start_clip + random.random() * num_sec / self.num_frames
        else:
            fps = self.num_frames / num_sec

        video = get_video_frames(video_fp, start_clip, end_clip, self.num_frames, fps=fps)

        if self.output_for_visualization:
            vis_video = self.vis_transform(video)

        if self.transforms is not None:
            video = self.transforms(video)

        return {'video': video, 'text': caption, 'time': num_sec, 'dataset': self.dataset_name, 'path': rel_fp,
                'idx': str(idx),
                'vis_video': (vis_video if self.output_for_visualization else 0),
                'max_text_length': self.max_text_length,
                }