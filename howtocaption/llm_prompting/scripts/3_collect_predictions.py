import os.path
import pickle
import tqdm
import argparse
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--asr-path", type=str)
    parser.add_argument("--output-path", type=str, default=None)

    args = parser.parse_args()

    with open(args.config) as fin:
        config = yaml.safe_load(fin)

    word_blocks = config['word_blocks']
    save_dir = config['save_dir']

    exp_name = os.path.splitext(os.path.basename(args.config))[0]
    result_dir = os.path.join(save_dir, exp_name)

    with open(args.asr_path, 'rb') as fin:
        data = pickle.load(fin)

    output = {}
    for id_, (key, val) in enumerate(tqdm.tqdm(data.items())):
        path = f'{result_dir}/{key[0]}/{key[1]}/{key}.pickle'
        with open(path, 'rb') as fin:
            pred = pickle.load(fin)
            if len(pred) > 0 and isinstance(pred[0], list):
                pred = [x[0] for x in pred]
            output[key] = {
                'start': val[word_blocks]['start'],
                'end': val[word_blocks]['end'],
                'prediction': pred
            }

    output_path = args.output_path
    if output_path is None:
        output_path = os.path.join(save_dir, f'{exp_name}.pickle')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(output)
    with open(output_path, 'wb') as fout:
        pickle.dump(output, fout)



