import os
import pickle
import random

import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import ffmpeg
import zlib
import json

from howtocaption.data_loader.video_datasets.utils import get_video_frames
from howtocaption.data_loader.transforms import init_transform_dict


class HowTo100M(Dataset):
    # adopted from https://github.com/antoine77340/MIL-NCE_HowTo100M/blob/master/video_loader.py
    """HowTo100M Video-Text loader."""

    def __init__(
            self,
            csv,
            video_root,
            caption_path,
            split='train',
            num_frames=4,
            transforms=None,

            return_all_frames_1fps=False,
            lm_loss_weight=1,
            ita_loss_weight=1,
            dataset_name='HowTo100M',
            output_for_visualization=False,
            meta_info_path=None, #  optional path to dict {video_id: {'width': x, 'height': y}, ...}
            process_only_part_i=None,
            number_of_parts=None,
            captions_are_zipped=False,
            aggregate_to_min_time=False,
            min_time=8.0,
    ):
        """
        Args:
        """
        self.split = split
        self.video_root = video_root
        self.min_time = min_time
        self.num_frames = num_frames
        self.transforms = transforms
        self.return_all_frames_1fps = return_all_frames_1fps

        self.dataset_name = dataset_name

        self.lm_loss_weight = lm_loss_weight
        self.ita_loss_weight = ita_loss_weight
        self.aggregate_to_min_time = aggregate_to_min_time
        self.captions_are_zipped = captions_are_zipped

        assert isinstance(csv, str)
        assert isinstance(caption_path, str)
        self.csv = pd.read_csv(csv)

        if process_only_part_i is not None:
            assert number_of_parts is not None
            size = int(np.ceil(len(self.csv) / number_of_parts))
            self.csv = self.csv[process_only_part_i * size: (process_only_part_i + 1) * size]

        with open(caption_path, 'rb') as fin:
            self.captions = pickle.load(fin)

        video_ids = set(self.csv['video_id']).intersection(self.captions.keys())
        self.csv = self.csv[self.csv['video_id'].isin(video_ids)]

        self.output_for_visualization = output_for_visualization
        self.vis_transform = init_transform_dict()['visualization']

        if meta_info_path is not None:
            with open(meta_info_path, 'rb') as fin:
                self.meta_info = pickle.load(fin)
        else:
            self.meta_info = None

    def __len__(self):
        return len(self.csv)

    def _find_nearest_candidates(self, captions, ind):
        start, end = ind, ind
        diff = captions['end'][end] - captions['start'][start]
        # Extend the video clip if shorter than the minimum desired clip duration
        offset_from_end = 1
        while diff < self.min_time:
            if start > 0 and end < len(captions['end']) - 1 - offset_from_end:
                d1 = captions['end'][end + 1] - captions['start'][start]
                d2 = captions['end'][end] - captions['start'][start - 1]
                # Use the closest neighboring video clip
                if d2 <= d1:
                    start -= 1
                else:
                    end += 1
            # If no video clips after it, use the clip before it
            elif start > 0:
                start -= 1
            # If no video clips before it, use the clip after it.
            elif end < len(captions['end']) - 1 - offset_from_end:
                end += 1
            # If there's no clips before or after
            else:
                break
            diff = captions['end'][end] - captions['start'][start]

        return start, end

    def _find_future_candidates(self, captions, ind):
        start, end = ind, ind
        diff = captions['end'][end] - captions['start'][start]
        # Extend the video clip if shorter than the minimum desired clip duration
        offset_from_end = 1
        while diff < self.min_time:
            if end < len(captions['end']) - 1 - offset_from_end:
                end += 1
            else:
                break
            diff = captions['end'][end] - captions['start'][start]
        return start, end

    def _sample_clip_ids(self, captions):
        if self.aggregate_to_min_time:
            offset_from_end = 1
            ind = random.randint(0, len(captions['text']) - 1 - offset_from_end) if len(captions['text']) - offset_from_end > 0 else 0

            start_id, end_id = self._find_nearest_candidates(captions, ind)
            return start_id, end_id
        else:
            ind = random.randint(0, len(captions['text']) - 1)  # not the same as np.random.randint
            return ind, ind

    def _get_text(self, captions, start_id, end_id):
        if start_id == end_id:
            texts = captions['text'][start_id]
            if isinstance(texts, list):
                cur_text = np.random.choice(texts)
            else:
                cur_text = texts
        else:
            cur_text = '. '.join(captions['text'][start_id:end_id + 1])
        return cur_text

    def __getitem__(self, idx):
        video_id = self.csv['video_id'].iloc[idx]
        video_path = self.csv['video_path'].iloc[idx]
        video_path = os.path.join(self.video_root, video_path)

        captions = self.captions[video_id]

        if self.captions_are_zipped:
            captions = zlib.decompress(captions)
            captions = json.loads(captions)

        # read width and height from meta to speed up video reading (avoid ffmpeg.probe)
        if self.meta_info is not None:
            if video_id in self.meta_info:
                width = self.meta_info[video_id]['width']
                height = self.meta_info[video_id]['height']
            else:
                print("Meta info is not found for id", video_id, flush=True)
                width = None
                height = None
        else:
            width = None
            height = None

        if self.return_all_frames_1fps:
            try:
                probe = ffmpeg.probe(video_path)
                secs = int(np.floor(float(probe['format']['duration'])))
            except Exception as excep:
                secs = 1
                print("Warning: ffmpeg error. video path: {} error. Error: {}".format(video_path, excep), flush=True)

            video = get_video_frames(video_path, 0, secs, num_frames=max(1, secs), fps=1, width=width, height=height)

            if self.output_for_visualization:
                vis_video = self.vis_transform(video)

            if self.transforms is not None:
                video = self.transforms(video)

            return {'video': video, 'text': '',
                    'start_time': 0, 'end_time': secs,
                    'time': secs, 'dataset': self.dataset_name, 'video_id': video_id,
                    'vis_video': (vis_video if self.output_for_visualization else 0)}
        else:
            start_id, end_id = self._sample_clip_ids(captions)

            start_time, end_time = captions['start'][start_id], captions['end'][end_id]
            cur_text = self._get_text(captions, start_id, end_id)
            video = get_video_frames(video_path, start_time, end_time, self.num_frames,
                                     width=width, height=height, central_frames=True)

            output = {}
            if self.output_for_visualization:
                vis_video = self.vis_transform(video)
                output['vis_video'] = vis_video
                output['video_id'] = video_id
                output['start_id'] = start_id

            if self.transforms is not None:
                video = self.transforms(video)

            output.update({
                      'video': video,
                      'start_time': start_time, 'end_time': end_time,
                      'time': end_time - start_time,
                      'dataset': self.dataset_name,
                      'lm_loss_weight': self.lm_loss_weight,
                      'ita_loss_weight': self.ita_loss_weight,
                      })

            update_output = {'text': cur_text}
            output.update(update_output)
            return output

