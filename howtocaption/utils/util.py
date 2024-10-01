import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import yaml

def read_yaml(fname):
    with fname.open('rt') as handle:
        class OrderedLoader(yaml.SafeLoader):
            pass
        def construct_mapping(loader, node):
            loader.flatten_mapping(node)
            return OrderedDict(loader.construct_pairs(node))
        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            construct_mapping)
        return yaml.load(handle, OrderedLoader)


def write_yaml(content, fname):
    with fname.open('wt') as handle:
        class OrderedDumper(yaml.SafeDumper):
            pass
        def _dict_representer(dumper, data):
            return dumper.represent_mapping(
                yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                data.items())
        OrderedDumper.add_representer(OrderedDict, _dict_representer)
        return yaml.dump(content, handle, OrderedDumper)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def read_config(fname):
    if fname.suffix == '.json':
        return read_json(fname)
    elif fname.suffix == '.yaml':
        return read_yaml(fname)
    else:
        raise NotImplementedError

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, neptune_run=None):
        self.neptune_run = neptune_run
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1, step=None):
        if key not in self._data.index:
            new_row = pd.DataFrame(index=[key], columns=['total', 'counts', 'average'])
            for col in new_row.columns:
                new_row[col].values[:] = 0
            self._data = pd.concat((self._data, new_row))

        if self.neptune_run is not None:
            self.neptune_run[key].log(value, step=step)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


