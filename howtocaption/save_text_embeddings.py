import argparse
import torch
import numpy as np
from datetime import datetime
import tqdm
import pandas as pd
from sacred import Experiment

import os
import pickle
import re
import sys

import howtocaption.model as module_arch
from howtocaption.parse_config import ConfigParser
from howtocaption.base.base_trainer import fix_module_in_state_dict


ex = Experiment('save_text_embeddings', save_git_info=False)


@ex.main
def run():
    print(f"Config: {config['name']}")

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
    llm_prediction_name = os.path.splitext(os.path.basename(args.llm_predictions))[0]


    with open(args.llm_predictions, 'rb') as fin:
        llm_predictions = pickle.load(fin)

    if args.process_only_part_i is not None:
        assert args.number_of_parts is not None
        assert args.csv is not None

        csv = pd.read_csv(args.csv)
        size = int(np.ceil(len(csv) / args.number_of_parts))
        csv = csv[args.process_only_part_i * size: (args.process_only_part_i + 1) * size]
        allowed_video_ids = csv['video_path'].map(lambda x: os.path.splitext(os.path.basename(x))[0]).tolist()
        llm_predictions = {key: val for key, val in llm_predictions.items() if key in allowed_video_ids}

    def preprocess(text):
        replace_none = ['*', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.']
        for pat in replace_none:
            text = text.replace(pat, '')
        text = text.replace('\n', '.')
        return [sent.strip() for sent in text.split('.') if sent.strip() != '']

    add_info_dict = {}
    anno_dict = {}
    print('Preprocessing: parsing sentences and timestamps...', flush=True)
    for video_id, data in tqdm.tqdm(llm_predictions.items()):
        texts = data['prediction']
        global_starts = [np.floor(x) for x in data['start']]
        global_ends = [np.ceil(x) for x in data['end']]

        pat = re.compile('^(\d+)s:(.*)')
        anno_dict[video_id] = []
        num_sents = []
        start = []
        end = []
        for text, global_start, global_end in zip(texts, global_starts, global_ends):
            text = preprocess(text)

            for sent in text:
                match = pat.match(sent)

                if match is not None:
                    s = int(match.group(1))
                    e = s + 8
                    sent = match.group(2).strip()

                    # exclude captions that were predicted outside original start end time:
                    if not((global_start <= s) and (s <= global_end)):
                        continue

                    anno_dict[video_id].append(sent)
                    start.append(s)
                    end.append(e)
                    num_sents.append(1)

            add_info_dict[video_id] = {
                'num_sents': num_sents,
                'start': start,
                'end': end,
            }

    output = {}
    missed_videos = 0
    counter = 0

    anno_dict = list(anno_dict.items())
    print("Found unique videos: ", len(anno_dict), flush=True)
    print("Found number of captions: ", sum(len(x[1]) for x in anno_dict), flush=True)

    with torch.no_grad():
        for video_id, captions in tqdm.tqdm(anno_dict):
            try:
                text_features = model.encode_text(captions)
                counter += 1
            except Exception as e:
                print(e, file=sys.stderr)
                missed_videos += 1
                continue
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.detach().cpu().numpy()
            output[video_id] = {'text': captions, 'features': text_features}

            if len(add_info_dict) != 0:
                output[video_id].update(add_info_dict[video_id])

            if counter % 1000 == 0:
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                print(current_time, video_id, flush=True)

        if args.process_only_part_i is not None:
            path = os.path.join(save_dir,  f'text_{config_name}_{llm_prediction_name}_part{args.process_only_part_i}.pickle')
        else:
            path = os.path.join(save_dir,  f'text_{config_name}_{llm_prediction_name}.pickle')

        print(f"Saving results into {path}")
        save_results(output, path)
        print("Missed videos", missed_videos)


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fout:
        pickle.dump(results, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('--device', default='cuda')

    parser.add_argument('--llm_predictions', default=None, type=str)

    # for processing only part of original data
    parser.add_argument('--csv', default=None, type=str)
    parser.add_argument('--process_only_part_i', default=None, type=int)
    parser.add_argument('--number_of_parts', default=None, type=int)

    config = ConfigParser(parser, test=True)
    args = config.args
    ex.add_config(config.config)

    ex.run()
