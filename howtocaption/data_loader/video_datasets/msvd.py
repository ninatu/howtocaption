import os
import random
import numpy as np
from torch.utils.data import Dataset
import ffmpeg
import pickle

from howtocaption.data_loader.video_datasets.utils import get_video_frames
from howtocaption.data_loader.transforms import init_transform_dict


class MSVD(Dataset):
    def __init__(self, data_root, split, num_frames=4, transforms=None,
                 dataset_name='MSVD',
                 multi_sentence_per_video=True,
                 output_for_visualization=False,
                 max_text_length=-1,
                 sample_per_video=False
                 ):
        super(MSVD, self).__init__()

        self.data_root = data_root
        self.multi_sentence_per_video = multi_sentence_per_video
        self._load_metadata(split)
        self.split = split

        self.transforms = transforms
        self.num_frames = num_frames

        self.dataset_name = dataset_name
        self.output_for_visualization = output_for_visualization
        self.max_text_length = max_text_length
        if output_for_visualization:
            self.vis_transform = init_transform_dict()['visualization']
        self.sample_per_video = sample_per_video

    def _load_metadata(self, split):
        assert split in ["train", "val", "test"]
        video_id_path_dict = {}
        if split == 'train':
            video_id_path_dict["train"] = os.path.join(self.data_root, 'msvd_data', "train_list.txt")
        elif split == 'val':
            video_id_path_dict["val"] = os.path.join(self.data_root, 'msvd_data', "val_list.txt")
        else:
            video_id_path_dict["test"] = os.path.join(self.data_root, 'msvd_data', "test_list.txt")

        caption_file = os.path.join(self.data_root, 'msvd_data', "raw-captions.pkl")

        with open(video_id_path_dict[split], 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]

        with open(caption_file, 'rb') as f:
            captions = pickle.load(f)
        self.captions = captions

        video_dict = {}
        features_path = os.path.join(self.data_root, 'YouTubeClips')
        for root, dub_dir, video_files in os.walk(features_path):
            for video_file in video_files:
                video_id_ = ".".join(video_file.split(".")[:-1])
                if video_id_ not in video_ids:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_
        self.video_dict = video_dict
        self.video_ids = video_ids

        self.sentences_dict = {}
        self.cut_off_points = []
        self.video_ids2video_idx = {}
        for idx, video_id in enumerate(video_ids):
            self.video_ids2video_idx[video_id] = idx
            assert video_id in captions
            for cap in captions[video_id]:
                cap_txt = " ".join(cap)
                self.sentences_dict[len(self.sentences_dict)] = (video_id, cap_txt)
            self.cut_off_points.append(len(self.sentences_dict))

        self.multi_sentence_per_video = self.multi_sentence_per_video    # !!! important tag for eval
        if split == "val" or split == "test":
            self.sentence_num = len(self.sentences_dict)
            self.video_num = len(video_ids)
            assert len(self.cut_off_points) == self.video_num
            print("For {}, sentence number: {}".format(split, self.sentence_num))
            print("For {}, video number: {}".format(split, self.video_num))

        print("Video number: {}".format(len(self.video_dict)))
        print("Total Paire: {}".format(len(self.sentences_dict)))
        self.sample_len = len(self.sentences_dict)

    def get_full_caption(self, idx):
        if isinstance(idx, int):
            video_id = self.video_ids[idx]
        else:
            video_id = idx
        return [' '.join(itm) for itm in self.captions[video_id]]

    def __len__(self):
        if self.sample_per_video:
            if self.split == 'train':
                return len( self.video_ids)
            else:
                raise NotImplementedError
        else:
            if self.split == 'train':
                return self.sample_len
            return len(self.video_dict)

    def __getitem__(self, idx):
        if self.sample_per_video:
            if self.split == 'train':
                video_id = self.video_ids[idx]
                video_fp = self.video_dict[video_id]
                caption = " ".join(np.random.choice(self.captions[video_id]))
            else:
                raise NotImplementedError
        else:
            if self.split == 'train':
                video_id, caption = self.sentences_dict[idx]
                video_fp = self.video_dict[video_id]
            else:
                video_id = self.video_ids[idx]
                video_fp = self.video_dict[video_id]
                caption = " ".join(self.captions[video_id][0])

        rel_fp = f'{video_id}.avi'
        asr = ''

        assert isinstance(caption, str)
        probe = ffmpeg.probe(video_fp)

        start_clip = 0
        end_clip = np.floor(float(probe['format']['duration']))

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

        output = {'video': video, 'asr': asr, 'text': caption, 'time': num_sec, 'dataset': self.dataset_name, 'path': rel_fp, 'idx': video_id,
                  'video_numerical_idx': self.video_ids2video_idx[video_id],
                  'vis_video': (vis_video if self.output_for_visualization else 0),
                  'max_text_length': self.max_text_length,
                  }
        return output
