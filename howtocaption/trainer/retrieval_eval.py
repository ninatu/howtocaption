import itertools

import numpy as np
import scipy.stats


def retrieval_metrics(sims, break_ties='averaging', complete_dataset_size=None):
    num_queries, num_vids = sims.shape
    if complete_dataset_size is not None:
        num_queries = complete_dataset_size

    sx = np.sort(-sims, axis=1)
    d = np.diag(-sims)
    d = d[:, np.newaxis]
    diff = sx - d
    if break_ties == 'optimistically':
        ind = np.argmax(diff == 0, axis=1)
    elif break_ties == 'averaging':
        locs = np.argwhere(diff == 0)
        grouped_locs = [list(values) for n_row, values in itertools.groupby(locs, key=lambda x: x[0])]
        ind = [np.mean(list(map(lambda x: x[1], locs))) for locs in grouped_locs]
        ind = np.array(ind)
    else:
        raise NotImplementedError
    return cols2metrics(ind, num_queries)


def cols2metrics(cols, num_queries):
    metrics = {}
    metrics["R1"] = 100 * float(np.sum(cols == 0)) / num_queries
    metrics["R5"] = 100 * float(np.sum(cols < 5)) / num_queries
    metrics["R10"] = 100 * float(np.sum(cols < 10)) / num_queries
    metrics["R50"] = 100 * float(np.sum(cols < 50)) / num_queries
    metrics["MedR"] = np.median(cols) + 1
    metrics["MeanR"] = np.mean(cols) + 1
    stats = [metrics[x] for x in ("R1", "R5", "R10")]
    metrics["geometric_mean_R1-R5-R10"] = scipy.stats.mstats.gmean(stats)
    return metrics