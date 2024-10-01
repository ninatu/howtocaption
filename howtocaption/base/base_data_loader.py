import torch
from torch.utils.data import DataLoader
from howtocaption.utils.dist_utils import is_dist_avail_and_initialized, get_world_size, get_rank


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, num_workers, drop_last, collate_fn=None):
        if is_dist_avail_and_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                      num_replicas=get_world_size(),
                                                                      rank=get_rank(),
                                                                      shuffle=shuffle,
                                                                      drop_last=drop_last)
            shuffle = False
        else:
            sampler = None

        super().__init__(dataset, batch_size, shuffle, sampler=sampler, num_workers=num_workers, collate_fn=collate_fn,
                         pin_memory=True, drop_last=drop_last)
