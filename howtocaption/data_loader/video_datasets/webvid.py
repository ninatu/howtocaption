import os
import pandas as pd
from torch.utils.data import Dataset

from howtocaption.data_loader.video_datasets.utils import get_video_frames


class WebVid2M(Dataset):
    def __init__(self, data_root, split, num_frames=4, transforms=None,
                 dataset_name='WebVid2M',
                 sample_beginning=False,
                 central_frames=False,
                 ):
        super(WebVid2M, self).__init__()

        self.data_root = data_root
        self._load_metadata(split)
        self.split = split

        self.transforms = transforms
        self.num_frames = num_frames

        self.dataset_name = dataset_name
        self.sample_beginning = sample_beginning
        self.central_frames = central_frames

    def _load_metadata(self, split):
        assert split in ["train"]
        self.csv = pd.read_csv(os.path.join(self.data_root, f'results_2M_{split}_downloaded.csv'))

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        data = self.csv.iloc[idx]

        video_id = data["videoid"]
        rel_fp = f'videos/{data["page_dir"]}/{data["videoid"]}.mp4'
        video_fp = os.path.join(self.data_root, rel_fp)
        caption = data['name']

        video = get_video_frames(video_fp, start=0, end=None, num_frames=self.num_frames,
                                sample_beginning=self.sample_beginning, central_frames=self.central_frames)

        if self.transforms is not None:
            video = self.transforms(video)

        output = {'video': video,  'text': caption, 'time': 0,
                  'dataset': self.dataset_name}
        output['start_time'] = 0
        output['end_time'] = 0

        return output
