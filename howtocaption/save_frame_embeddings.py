import argparse
import torch
import numpy as np
from datetime import datetime
import tqdm
import os
import pickle
from sacred import Experiment

import howtocaption.data_loader as module_data
import howtocaption.model as module_arch

from howtocaption.trainer.vl_trainer import _move_to_device
from howtocaption.parse_config import ConfigParser
from howtocaption.train import init_dataloaders
from howtocaption.base.base_trainer import fix_module_in_state_dict

ex = Experiment('predict', save_git_info=False)


@ex.main
def run():
    print(f"Config: {config['name']}")

    print("Creating dataset")
    print("Setting batch_size to 1")
    config['data_loader']['args']['batch_size'] = 1

    data_loader = init_dataloaders(config, 'data_loader', module_data,
                                         process_only_part_i=args.process_only_part_i,
                                         number_of_parts=args.number_of_parts)

    print("Creating model")
    model = config.initialize('arch', module_arch)

    if config['load_weights'] is not None:
        checkpoint = torch.load(config['load_weights'])
        print("Loading model weights: {} ...".format(config['load_weights']), flush=True)
        print("from epoch: {} ...".format(checkpoint['epoch']), flush=True)
        state_dict = checkpoint['state_dict']
        state_dict = fix_module_in_state_dict(state_dict, model)
        model.load_state_dict(state_dict)
    model.eval()
    device = torch.device(args.device)
    model = model.to(device)

    save_dir = config['save_dir']
    config_name = config["name"]

    with torch.no_grad():
        for dl_idx, dl in enumerate(data_loader):
            output = {}
            for data_idx, data in tqdm.tqdm(enumerate(dl)):
                assert data['video'].shape[0] == 1 # batch size = 1
                assert len(data['video_id']) == 1
                video_id = data['video_id'][0]
                video = data['video']

                cur_size = video.size()
                frames = video.view(cur_size[0] * cur_size[1], *cur_size[2:])

                output_embed = []
                max_batch_size = args.batch_size
                for offset_i in range(int(np.ceil(frames.size(0) / max_batch_size))):
                    cur_frames = frames[offset_i * max_batch_size:(offset_i + 1) * max_batch_size]
                    cur_frames = _move_to_device(cur_frames, device)

                    cur_embed = model.encode_image(cur_frames[None])
                    cur_embed /= cur_embed.norm(dim=-1, keepdim=True)
                    output_embed.append(cur_embed)
                output_embed = torch.cat(output_embed, dim=0)

                output[video_id] = {'start': data['start_time'][0].item(), 'end': data['end_time'][0].item(),
                                    'frames': output_embed.detach().cpu().numpy().astype('float16')}

                if data_idx % 100 == 0:
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    print(f'{current_time}: batch', data_idx, flush=True)

    if args.process_only_part_i is not None:
        path = os.path.join(save_dir, f'video_{config_name}_part{args.process_only_part_i}.pickle')
    else:
        path = os.path.join(save_dir, f'video_{config_name}.pickle')
    print(f"Saving results into {path}")
    save_results(output, path)


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fout:
        pickle.dump(results, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--process_only_part_i', default=None, type=int)
    parser.add_argument('--number_of_parts', default=None, type=int)

    config = ConfigParser(parser, test=True)
    args = config.args

    ex.add_config(config.config)

    ex.run()
