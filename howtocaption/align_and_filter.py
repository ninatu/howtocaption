import pickle
import torch
from collections import defaultdict, Counter
import tqdm
import os
import numpy as np
import argparse
import copy
import json
import zlib


def run(args):
    THRESHOLD = args.threshold
    TOP_QUANTILE_THRESHOLD = args.top_quantile_threshold
    TOP_PAIRS_THRESHOLD = args.top_pairs_threshold
    OFFSET_SECS = args.offset_secs
    SECS = args.secs

    print("Loading text embeddings ...")
    PRELOADED_VIDEO_FEATURES = []
    for frame_embeddings in args.frame_embeddings:
        with open(frame_embeddings, 'rb') as fin:
            PRELOADED_VIDEO_FEATURES.append(pickle.load(fin))

    print("Loading text embeddings ...")
    PRELOADED_TEXT_FEATURES = []
    for text_embeddings in args.text_embeddings:
        with open(text_embeddings, 'rb') as fin:
            PRELOADED_TEXT_FEATURES.append(pickle.load(fin))
    assert len(PRELOADED_VIDEO_FEATURES) == len(PRELOADED_TEXT_FEATURES)

    def find_alignment(video_ids, threshold):
        output = []
        found = 0
        selected_how_to_indices = []

        for video_id in tqdm.tqdm(video_ids):
            try:
                if any((video_id not in features) for features in PRELOADED_VIDEO_FEATURES):
                    print(f"Video features {video_id} are missing", flush=True)
                    continue

                N_features = len(PRELOADED_VIDEO_FEATURES)
                RIGHT_PAD = int(np.ceil(SECS / 2.0))

                cur_video_embeds_list = []
                cur_text_embeds_list = []
                for i in range(N_features):
                    cur_video_embeds = PRELOADED_VIDEO_FEATURES[i][video_id]['frames']
                    cur_text_embeds = PRELOADED_TEXT_FEATURES[i][video_id]['features']
                    if isinstance(cur_text_embeds, np.ndarray):
                        cur_text_embeds = torch.from_numpy(cur_text_embeds.astype(np.float32))
                    cur_video_embeds = torch.from_numpy(cur_video_embeds.astype(np.float32))
                    cur_text_embeds = cur_text_embeds.cuda()
                    cur_video_embeds = cur_video_embeds.cuda()

                    # get clip embeddings (average over frame embeddings)
                    cur_video_embeds = cur_video_embeds[None]
                    cur_video_embeds = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(cur_video_embeds, (0, 0, RIGHT_PAD, SECS - RIGHT_PAD), mode='reflect'), (SECS, 1), stride=1)[0]
                    cur_video_embeds_list.append(cur_video_embeds)
                    cur_text_embeds_list.append(cur_text_embeds)

                cur_texts = PRELOADED_TEXT_FEATURES[0][video_id]['text']
                starts = PRELOADED_TEXT_FEATURES[0][video_id]['start']
                ends = PRELOADED_TEXT_FEATURES[0][video_id]['end']
                num_sents = PRELOADED_TEXT_FEATURES[0][video_id]['num_sents']
                cur_offset = 0
                for cur_i in range(len(starts)):
                    start = starts[cur_i]
                    end = ends[cur_i]

                    # find center of the segment
                    start = int(np.floor((start + end) / 2))
                    end = start + 1

                    # apply offset
                    start = max(0, int(np.floor(start)) - OFFSET_SECS)
                    end = int(np.ceil(end)) + OFFSET_SECS
                    is_empty = False
                    sims = 0
                    for i in range(N_features):
                        cur_video_embeds = cur_video_embeds_list[i]
                        cur_text_embeds = cur_text_embeds_list[i]
                        video_embeds = cur_video_embeds[start:end]
                        if len(video_embeds) == 0:
                            is_empty = True
                            break
                        text_embeds = cur_text_embeds[cur_offset:cur_offset + num_sents[cur_i]]

                        sims = text_embeds @ video_embeds.t() + sims
                    if is_empty:
                        continue

                    texts = cur_texts[cur_offset:cur_offset + num_sents[cur_i]]
                    cur_offset += num_sents[cur_i]
                    values, cur_indices = torch.topk(sims, k=1, dim=1)
                    values, cur_indices = values[:, -1], cur_indices[:, -1]
                    if threshold is not None:
                        cur_indices[values < threshold] = -1
                    cur_indices = cur_indices.cpu().tolist()

                    for ind, text, similarity in zip(cur_indices, texts, values):
                        if ind != -1:
                            start_time = max(0, start + ind - RIGHT_PAD)
                            end_time = start + ind + (SECS - RIGHT_PAD)
                            output.append((video_id, start_time, end_time, text, similarity))
                            found += 1
                            selected_how_to_indices.append(f'{video_id}_{start_time}')
            except Exception as excep:
                print("Error: {}".format(excep), flush=True)

        return found, output, selected_how_to_indices

    all_video_ids = list(PRELOADED_TEXT_FEATURES[0].keys())
    print(f"Number of video_ids in text dict {len(all_video_ids)}")
    print(f"Number of video_ids in video dict: {len( list(PRELOADED_VIDEO_FEATURES[0].keys()))}")

    print("Creating alignment ...", flush=True)

    if TOP_PAIRS_THRESHOLD is not None:
        print("top_pairs_threshold is defined --> estimating top_quantile_threshold ...", flush=True)
        video_ids = copy.deepcopy(all_video_ids)
        np.random.shuffle(video_ids)
        n_clips = sum(len(PRELOADED_TEXT_FEATURES[0][video_id]['text']) for video_id in video_ids)
        if args.number_of_parts is not None:
            TOP_QUANTILE_THRESHOLD = (TOP_PAIRS_THRESHOLD / args.number_of_parts) / n_clips
        else:
            TOP_QUANTILE_THRESHOLD = TOP_PAIRS_THRESHOLD / n_clips
        print(f"Estimated top_quantile_threshold is {TOP_QUANTILE_THRESHOLD}", flush=True)

    if TOP_QUANTILE_THRESHOLD is not None:
        print("top_quantile_threshold is defined --> estimating threshold ...", flush=True)
        video_ids = copy.deepcopy(all_video_ids)
        np.random.shuffle(video_ids)
        n_random_videos = min(5000, len(video_ids))
        print(f"Estimate threshold on {n_random_videos} random videos", flush=True)
        found, output, selected_how_to_indices = find_alignment(video_ids[:n_random_videos], None)
        similarities = [similarity.cpu().item() for video_id, start_time, end_time, text, similarity in output]
        THRESHOLD = np.quantile(similarities, (1 - TOP_QUANTILE_THRESHOLD))
        print(f"Estimated threshold is {THRESHOLD} ", flush=True)

    print(f"Alignment and filtering using threshold {THRESHOLD}", flush=True)
    found, output, selected_how_to_indices = find_alignment(all_video_ids, THRESHOLD)
    print(f"Alignment and filtering is finished!", flush=True)

    print(f"Only {found} text-video clip pairs are left")
    print("Number of unique video clips: ", len(Counter(selected_how_to_indices)))

    ######## ----- SAVING  --------

    new_data = defaultdict(lambda: {'start': [], 'end': [], 'text': []})

    for video_id, start_time, end_time, text, similatiry in tqdm.tqdm(output):
        new_data[video_id]["start"].append(start_time)
        new_data[video_id]["end"].append(end_time)
        if args.with_scores:
            new_data[video_id]["text"].append([(text, similatiry.item())])
        else:
            new_data[video_id]["text"].append([text])

    print(f"Saving alignments to {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as fout:
        pickle.dump(dict(new_data), fout)

    name, ext = os.path.splitext(args.output)
    zipped_path = f'{name}_zipped{ext}'
    print(f"Zipping alignments to {zipped_path}")
    zipped_data = {}
    for key, val in new_data.items():
        val = json.dumps(val)
        val = zlib.compress(val.encode(), level=9)
        zipped_data[key] = val

    with open(zipped_path, 'wb') as fout:
        pickle.dump(zipped_data, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--frame_embeddings', type=str, nargs='+')
    parser.add_argument('--text_embeddings', type=str, nargs='+')
    parser.add_argument('--output', type=str)

    parser.add_argument('--threshold', default=None, type=float)
    parser.add_argument('--top_quantile_threshold', default=None, type=float)
    parser.add_argument('--top_pairs_threshold', default=None, type=int)

    parser.add_argument('--offset_secs', default=10, type=int)
    parser.add_argument('--secs', default=8, type=int)

    parser.add_argument('--process_only_part_i', default=None, type=str)
    parser.add_argument('--number_of_parts', default=None, type=int)
    parser.add_argument('--with_scores', default=0, type=int)

    args = parser.parse_args()
    assert sum([int(args.threshold is not None),
                int(args.top_quantile_threshold is not None),
                int(args.top_pairs_threshold is not None)]) <= 1
    run(args)