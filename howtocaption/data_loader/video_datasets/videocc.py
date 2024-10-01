import os
import pandas as pd
from torch.utils.data import Dataset
import json

from howtocaption.data_loader.video_datasets.utils import get_video_frames


class VideoCC3M(Dataset):
    def __init__(self, data_root, split, num_frames=4, transforms=None,
                 dataset_name='VideoCC3M',
                 sample_beginning=False,
                 central_frames=False,
                 ):
        super(VideoCC3M, self).__init__()

        self.data_root = data_root
        self._load_metadata(split)
        self.split = split

        self.transforms = transforms
        self.num_frames = num_frames
        self.dataset_name = dataset_name

        self.sample_beginning = sample_beginning
        self.central_frames = central_frames

    def _load_metadata(self, split):
        assert split == "train"
        self.csv = pd.read_csv(os.path.join(self.data_root, f'video_cc_public_downloaded.csv'))

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        data = self.csv.iloc[idx]

        video_id = data["video_id"]
        rel_fp = f'dataset/{data["page_dir"]:05d}/{data["video_id"]}.mp4'
        video_fp = os.path.join(self.data_root, rel_fp)
        caption = data['caption']

        try:
            json_path = f'dataset/{data["page_dir"]:05d}/{data["video_id"]}.json'
            with open(f'data/videocc3m/{json_path}') as fin:
                metadata = json.load(fin)
            width = metadata['video_metadata']['streams'][0]['width']
            height = metadata['video_metadata']['streams'][0]['height']
            clips = metadata['clips']
            assert len(clips) == 1
            end = clips[0][1] - clips[0][0]
        except Exception as excep:
            print("Warning:  video path: {} error. Error: {}".format(video_fp, excep), flush=True)
            width, height, end = None, None, None

        video = get_video_frames(video_fp, start=0, end=end, num_frames=self.num_frames,
                                sample_beginning=self.sample_beginning, central_frames=self.central_frames,
                                width=width, height=height)

        if self.transforms is not None:
            video = self.transforms(video)

        output = {'video': video, 'text': caption, 'time': 0, 'dataset': self.dataset_name, 'path': rel_fp, 'idx': video_id}

        return output
