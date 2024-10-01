import os
import random
import numpy as np
from torch.utils.data import Dataset
import ffmpeg

from howtocaption.data_loader.video_datasets.utils import get_video_frames
from howtocaption.data_loader.transforms import init_transform_dict


class LSMDC(Dataset):
    def __init__(self, data_root, split, num_frames=4, transforms=None,
                 dataset_name='LSDMC',
                 output_for_visualization=False,
                 max_text_length=-1
                 ):
        super(LSMDC, self).__init__()

        self.data_root = data_root
        self._load_metadata(split)
        self.split = split

        self.transforms = transforms
        self.num_frames = num_frames
        self.dataset_name = dataset_name
        self.output_for_visualization = output_for_visualization
        self.max_text_length = max_text_length
        if output_for_visualization:
            self.vis_transform = init_transform_dict()['visualization']

    def _load_metadata(self, split):
        assert split in ["train", "val", "test"]
        video_json_path_dict = {}
        if split == 'train':
            video_json_path_dict["train"] = os.path.join(self.data_root, 'lsmdc2016', "LSMDC16_annos_training.csv")
        elif split == 'val':
            video_json_path_dict["val"] = os.path.join(self.data_root, 'lsmdc2016', "LSMDC16_annos_val.csv")
        else:
            video_json_path_dict["test"] = os.path.join(self.data_root, 'lsmdc2016', "LSMDC16_challenge_1000_publictect.csv")

        # <CLIP_ID>\t<START_ALIGNED>\t<END_ALIGNED>\t<START_EXTRACTED>\t<END_EXTRACTED>\t<SENTENCE>
        # <CLIP_ID> is not a unique identifier, i.e. the same <CLIP_ID> can be associated with multiple sentences.
        # However, LSMDC16_challenge_1000_publictect.csv has no repeat instances
        video_id_list = []
        caption_dict = {}
        with open(video_json_path_dict[split], 'r') as fp:
            for line in fp:
                line = line.strip()
                line_split = line.split("\t")
                assert len(line_split) == 6
                clip_id, start_aligned, end_aligned, start_extracted, end_extracted, sentence = line_split
                caption_dict[clip_id] = {
                    'start': start_aligned,
                    'end': end_aligned,
                    'text': sentence,
                    'clip_id': clip_id
                }
                if clip_id not in video_id_list: video_id_list.append(clip_id)

        self.caption_dict = caption_dict

        features_path = os.path.join(self.data_root, 'avi')
        features_path2 = os.path.join(self.data_root, 'avi-m-vad-aligned')
        video_dict = {}
        for root, dub_dir, video_files in os.walk(features_path):
            for video_file in video_files:
                video_id_ = ".".join(video_file.split(".")[:-1])
                if video_id_ not in video_id_list:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_

        for root, dub_dir, video_files in os.walk(features_path2):
            for video_file in video_files:
                video_id_ = ".".join(video_file.split(".")[:-1])
                if video_id_ not in video_id_list:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_

        self.video_dict = video_dict

        # Get all captions
        self.iter2video_pairs_dict = {}
        for v in caption_dict.values():
            clip_id = v['clip_id']
            if clip_id not in self.video_dict:
                continue
            self.iter2video_pairs_dict[len(self.iter2video_pairs_dict)] = clip_id

        if split == 'test':
            assert len(self.iter2video_pairs_dict) == 1000

    def get_full_caption(self, idx):
        video_id = self.iter2video_pairs_dict[idx]
        return self.caption_dict[video_id]['text']

    def __len__(self):
        return len(self.iter2video_pairs_dict)

    def __getitem__(self, idx):
        video_id = self.iter2video_pairs_dict[idx]
        video_fp = self.video_dict[video_id]
        rel_fp = video_id
        asr = ''
        caption = self.caption_dict[video_id]['text']

        start_clip = 0
        probe = ffmpeg.probe(video_fp)
        end_clip = np.floor(float(probe['format']['duration']))

        num_sec = end_clip - start_clip - 0.1  # just in case
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

        output = {'video': video, 'asr': asr, 'text': caption, 'time': num_sec, 'dataset': self.dataset_name, 'path': rel_fp, 'idx': video_id,
                  'vis_video': (vis_video if self.output_for_visualization else 0),
                  'max_text_length': self.max_text_length,
                  }

        return output
