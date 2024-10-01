import pickle
import tqdm
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr", type=str, default='data/howto100m/asr_filtered.pickle')
    parser.add_argument("--n_words_max", type=int, default=200)
    parser.add_argument("--output_key", type=str, default='200w')

    args = parser.parse_args()
    n_words_max = args.n_words_max

    with open(args.asr, 'rb') as fin:
        data = pickle.load(fin)

    output = {}
    for name, x in tqdm.tqdm(data.items()):
        sentences = [sent for sent in x['text']]

        blocks = []
        starts = []
        ends = []
        all_words = 0

        cur_block = ''
        cur_words = 0
        cur_start = 1000000000
        cur_end = None

        for start, end, sent in zip(x['start'], x['end'], sentences):
            if len(sent.split(' ')) + cur_words <= n_words_max:
                cur_block += f'\n{int(start)}s: ' + sent

                cur_words += len(sent.split(' '))
                cur_start = min(cur_start, start)
                cur_end = end
            else:
                if cur_block != '':
                    blocks.append(cur_block)
                    starts.append(cur_start)
                    ends.append(cur_end)

                cur_block = sent
                cur_words = len(sent.split(' '))
                cur_start = start
                cur_end = end

        if cur_block != '':
            blocks.append(cur_block)
            starts.append(cur_start)
            ends.append(cur_end)

        x[args.output_key] = {
            'text': blocks,
            'start': starts,
            'end': ends,
        }

    with open(args.asr, 'wb') as fout:
        pickle.dump(data, fout)
