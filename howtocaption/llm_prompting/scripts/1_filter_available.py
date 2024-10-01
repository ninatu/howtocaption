import pickle
import json
import pandas as pd
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr", type=str, default='data/howto100m/sentencified_htm_1200k.json')
    parser.add_argument("--csv", type=str, default='data/howto100m/video_path_downloaded.csv')
    parser.add_argument("--output_folder", type=str, default='data/howto100m')

    args = parser.parse_args()
    csv = pd.read_csv(args.csv)

    with open(args.asr, 'r') as fin:
        asrs = json.load(fin)

    video_ids = set(csv['video_id']).intersection(asrs.keys())

    csv_available = csv[csv['video_id'].isin(video_ids)]
    asrs = {key: asrs[key] for key in video_ids}

    csv_debug = csv_available.iloc[:50]
    asrs_debug = {key: asrs[key] for key in csv_debug['video_id']}

    csv_available.to_csv(os.path.join(args.output_folder, 'video_path_filtered.csv'))
    csv_debug.to_csv(os.path.join(args.output_folder, 'video_path_filtered_s50.csv'))

    with open(os.path.join(args.output_folder, 'asr_filtered.pickle'), 'wb') as fout:
        pickle.dump(asrs, fout)

    with open(os.path.join(args.output_folder, 'asr_filtered_s50.pickle'), 'wb') as fout:
        pickle.dump(asrs_debug, fout)
