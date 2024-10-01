from torch.utils.data import Dataset
import torch

from torch.utils.data import DataLoader

from howtocaption.data_loader import video_datasets
from howtocaption.data_loader.transforms import init_transform_dict
from howtocaption.utils.dist_utils import get_rank, get_world_size


class VideoDataLoader(DataLoader):
    def __init__(self, dataset_type, dataset_args, batch_size, num_workers,
                 split='train', transform=None, shuffle=None, transform_params={},
                 prefetch_factor=2,
                 pin_memory=True,
                 **kwargs):
        if shuffle is None:
            shuffle = (split == 'train')

        if transform is None:
            transform = split
        transforms = init_transform_dict(**transform_params)[transform]
        dataset = getattr(video_datasets, dataset_type)(transforms=transforms, split=split, **dataset_args)

        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=get_world_size(), rank=get_rank(),
                                                      shuffle=shuffle)

        super(VideoDataLoader, self).__init__(dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=sampler,
            shuffle=False,
            collate_fn=None,
            drop_last=(split == 'train'),
            prefetch_factor=prefetch_factor)
