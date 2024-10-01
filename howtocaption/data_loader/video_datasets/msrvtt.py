import os
import random

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import ffmpeg
import json

from howtocaption.data_loader.video_datasets.utils import get_video_frames
from howtocaption.data_loader.transforms import init_transform_dict


class MSRVTT(Dataset):
    def __init__(self, data_root, cut, split, num_frames=4, transforms=None,
                 dataset_name='MSRVTT',
                 output_for_visualization=False,
                 max_text_length=-1
                 ):
        super(MSRVTT, self).__init__()

        self.metadata = self._load_metadata(data_root, cut, split)
        self.data_root = data_root
        self.split = split

        self.transforms = transforms
        self.num_frames = num_frames
        self.dataset_name = dataset_name
        self.output_for_visualization = output_for_visualization
        self.max_text_length = max_text_length
        if output_for_visualization:
            self.vis_transform = init_transform_dict()['visualization']

    def _load_metadata(self, data_root, cut, split):
        json_fp = os.path.join(data_root, 'annotation', 'MSR_VTT.json')
        with open(json_fp, 'r') as fid:
            data = json.load(fid)
        df = pd.DataFrame(data['annotations'])

        split_dir = os.path.join(data_root, 'high-quality', 'structured-symlinks')
        js_test_cap_idx_path = None
        challenge_splits = {"val", "public_server_val", "public_server_test"}
        if cut == "miech":
            train_list_path = "train_list_miech.txt"
            test_list_path = "test_list_miech.txt"
        elif cut == "jsfusion":
            train_list_path = "train_list_jsfusion.txt"
            test_list_path = "val_list_jsfusion.txt"
            js_test_cap_idx_path = "jsfusion_val_caption_idx.pkl"
        elif cut in {"full-val", "full-test"}:
            train_list_path = "train_list_full.txt"
            if cut == "full-val":
                test_list_path = "val_list_full.txt"
            else:
                test_list_path = "test_list_full.txt"
        elif cut in challenge_splits:
            train_list_path = "train_list.txt"
            if cut == "val":
                test_list_path = f"{cut}_list.txt"
            else:
                test_list_path = f"{cut}.txt"
        else:
            msg = "unrecognised MSRVTT split: {}"
            raise ValueError(msg.format(cut))

        train_df = pd.read_csv(os.path.join(split_dir, train_list_path), names=['videoid'])
        test_df = pd.read_csv(os.path.join(split_dir, test_list_path), names=['videoid'])

        if split == 'train':
            df = df[df['image_id'].isin(train_df['videoid'])]
        else:
            df = df[df['image_id'].isin(test_df['videoid'])]

        metadata = df.groupby(['image_id'])['caption'].apply(list)

        # use specific caption idx's in jsfusion
        if js_test_cap_idx_path is not None and split != 'train':
            caps = pd.Series(np.load(os.path.join(split_dir, js_test_cap_idx_path), allow_pickle=True))
            new_res = pd.DataFrame({'caps': metadata, 'cap_idx': caps})
            new_res['test_caps'] = new_res.apply(lambda x: [x['caps'][x['cap_idx']]], axis=1)
            metadata = new_res['test_caps']

        metadata = pd.DataFrame({'captions': metadata})
        return metadata

    def _get_video_path(self, sample):
        return os.path.join(self.data_root, 'videos', 'all', sample.name + '.mp4'), sample.name + '.mp4'

    def _get_caption(self, sample):
        if self.split == 'train':
            caption = random.choice(sample['captions'])
        else:
            caption = sample['captions'][0]
        return caption

    def get_full_caption(self, idx):
        sample = self.metadata.iloc[idx]
        return sample['captions']

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        idx = idx % len(self.metadata)

        sample = self.metadata.iloc[idx]
        video_fp, rel_fp = self._get_video_path(sample)
        caption = self._get_caption(sample)

        if isinstance(caption, str):
            probe = ffmpeg.probe(video_fp)

            start_clip = 0
            end_clip = np.floor(float(probe['format']['duration']))
        else:
            caption, start_clip, end_clip = caption

        num_sec = end_clip - start_clip - 0.1 # just in case
        if self.split == 'train':
            # sample start
            fps = (self.num_frames + 1) / num_sec
            start_clip = random.random() * num_sec / self.num_frames
        else:
            fps = self.num_frames / num_sec

        video = get_video_frames(video_fp, start_clip, end_clip, self.num_frames, fps=fps)

        if self.output_for_visualization:
            vis_video = self.vis_transform(video)

        if self.transforms is not None:
            video = self.transforms(video)

        output = {'video': video, 'text': caption, 'time': num_sec, 'dataset': self.dataset_name, 'path': rel_fp, 'idx': sample.name,
                  'vis_video': (vis_video if self.output_for_visualization else 0),
                  'max_text_length': self.max_text_length,
                  }
        return output
