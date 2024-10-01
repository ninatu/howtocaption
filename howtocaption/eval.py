import argparse
import collections
import torch
import howtocaption.data_loader as module_data
import howtocaption.model as module_arch
import howtocaption.trainer as module_trainer
from howtocaption.parse_config import ConfigParser
from sacred import Experiment
from howtocaption.train import init_dataloaders

import neptune.new as neptune
from neptune.new.integrations.sacred import NeptuneObserver

from howtocaption.utils.dist_utils import init_distributed_mode, is_main_process, get_rank, get_world_size

ex = Experiment('eval', save_git_info=False)


@ex.main
def run():
    print(f"Config: {config['name']}")
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
    print(model)

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

    config.config['trainer']['args']['resume_only_model'] = True
    trainer = config.initialize('trainer', module_trainer,
                                model=model,
                               loss=None,
                               metrics=None,
                               optimizer=None,
                               neptune_run=neptune_run,
                               config=config,
                               device=device,
                               data_loader=train_data_loader,
                               valid_data_loader=valid_data_loader,
                               lr_scheduler=None,
                               model_without_ddp=model_without_ddp)
    if args.eval_retrieval:
        trainer._eval_retrieval()
    if args.eval_captioning:
        trainer._eval_nlp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-n', '--neptune', action='store_true',
                        help='Whether to observe (neptune)')
    parser.add_argument('--eval_retrieval', action='store_true', default=False)
    parser.add_argument('--eval_captioning', action='store_true', default=False)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=0, type=int)

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser(parser, options)
    args = config.args

    ex.add_config(config.config)

    init_distributed_mode(args=args)

    neptune_run = None
    if args.neptune and is_main_process():
        # delete this error if you have added your own neptune credentials neptune.ai
        raise ValueError
        api_token = ''
        project = ''

        neptune_run = neptune.init(
            project=project,
            api_token=api_token,
            source_files=['imp_videocap/**/*.py', '*.py']
        )

        ex.observers.append(NeptuneObserver(run=neptune_run))

    ex.run()
