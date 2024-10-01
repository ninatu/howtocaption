import argparse
import collections
import torch
import numpy as np
import random
import howtocaption.data_loader as module_data
import howtocaption.lr_scheduler as module_lr_scheduler
import howtocaption.model as module_arch
import howtocaption.trainer as module_trainer
from howtocaption.parse_config import ConfigParser
from sacred import Experiment
import warnings
import logging

import neptune.new as neptune
from neptune.new.integrations.sacred import NeptuneObserver

from howtocaption.utils.dist_utils import init_distributed_mode, is_main_process, get_rank, get_world_size

ex = Experiment('train', save_git_info=False)


@ex.main
def run():
    print(f"Config: {config['name']}")

    # fix random seeds for reproducibility
    if config['seed'] is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

        seed = config['seed'] + get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    # setup data_loader instances
    torch.multiprocessing.set_start_method('spawn', force=True)

    print("Creating dataset")
    train_data_loader = init_dataloaders(config, 'train_data_loader', module_data)

    if config.config.get('valid_data_loader', None) is not None:
        valid_data_loader = init_dataloaders(config, 'valid_data_loader', module_data)
    else:
        valid_data_loader = None

    # build model architecture, then print to console
    print("Creating model")
    model = config.initialize('arch', module_arch)
    # print(model)

    # prepare for (multi-device) GPU training
    device = torch.device(args.device)

    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])#, find_unused_parameters=True)
        model_without_ddp = model.module

    else:
        model = model.to(device)
        model_without_ddp = model

    # build optimizer, learning rate scheduler
    trainable_params = [param for name, param in model.named_parameters() if param.requires_grad and 'lora' not in name]
    optimizer = config.initialize('optimizer', torch.optim, params=trainable_params)

    if config.config.get('lr_scheduler') is not None:
        if config['trainer']['args'].get('lr_scheduler_update') == 'iter':
            len_epoch = config['trainer']['args'].get('len_epoch')
            if len_epoch is None:
                len_epoch = min([len(dl) for dl in train_data_loader])
            if config['lr_scheduler']['type'] == 'CosineAnnealingLR':
                config['lr_scheduler']['args']['T_max'] *= len_epoch
            else:
                raise NotImplementedError()
        lr_scheduler = config.initialize('lr_scheduler', module_lr_scheduler, optimizer=optimizer)
    else:
        lr_scheduler = None


    trainer = config.initialize('trainer', module_trainer,
                                model=model,
                               loss=None,
                               metrics=None,
                               optimizer=optimizer,
                               neptune_run=neptune_run,
                               config=config,
                               device=device,
                               data_loader=train_data_loader,
                               valid_data_loader=valid_data_loader,
                               lr_scheduler=lr_scheduler,
                               model_without_ddp=model_without_ddp)

    trainer.train()


def init_dataloaders(config, data_loader_name, module_data, **kwargs):
    def fix_args_wrt_world_size(args_to_fix):
        for param in ['dataset_size', 'batch_size']:
            if param in args_to_fix:
                args_to_fix[param] = int(args_to_fix[param] / get_world_size())

    if "type" in config[data_loader_name] and "args" in config[data_loader_name]:
        fix_args_wrt_world_size(config[data_loader_name]['args'])
        return [config.initialize(data_loader_name, module_data, **kwargs)]
    elif isinstance(config[data_loader_name], list):
        data_loaders = []
        for idx in range(len(config[data_loader_name])):
            fix_args_wrt_world_size(config[data_loader_name][idx]['args'])
            data_loaders.append(config.initialize(data_loader_name, module_data, index=idx, **kwargs))
        return data_loaders
    else:
        raise ValueError("Check data_loader config, not correct format.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-n', '--neptune', action='store_true',
                        help='Whether to observe (neptune)')
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=0, type=int)
    parser.add_argument('--neptune_mode', default='async', type=str)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
    ]
    config = ConfigParser(parser, options)
    args = config.args

    ex.add_config(config.config)

    init_distributed_mode(args=args)

    neptune_run = None
    if args.neptune and is_main_process():

        # for resume
        neptune_id_path = config.save_dir.parent / 'neptune_id.txt'
        if neptune_id_path.exists():
            with open(neptune_id_path, 'r') as fin:
                with_neptune_id = fin.readline().strip()
        else:
            with_neptune_id = None

        # delete this error if you have added your own neptune credentials neptune.ai
        raise ValueError
        api_token = ''
        project = ''

        neptune_run = neptune.init(
            project=project,
            api_token=api_token,
            source_files=['imp_videocap/**/*.py', '*.py'],
            with_id=with_neptune_id,
            mode=args.neptune_mode,
        )
        # save neptune id to be able to resume logging to the same id
        if not neptune_id_path.exists():
            neptune_id = neptune_run["sys/id"].fetch()
            with open(neptune_id_path, 'w') as fout:
                fout.write(neptune_id)

        logging.getLogger("neptune.new.internal.operation_processors.async_operation_processor").setLevel(
            logging.CRITICAL)
        ex.observers.append(NeptuneObserver(run=neptune_run))

    ex.run()
