import getopt
import math
import sys
import numpy as np
from tqdm import tqdm

from dtw import dtw_naive, dtw_sakoe_chiba
from utils.distance import abs_distance, euclidean_distance
from utils.io import read_input_1d, read_input_3d

class Config:
    algo_func = False
    algo_kwargs = {}
    dimensions  = False
    prunning = False
    show_help = False

def predict(train, test, algo_func, algo_kwargs, prunning):
    y_pred = []
    confusion_matrix = {}

    labels = np.unique([label for label, _ in train])
    for l1 in labels:
        confusion_matrix[l1] = {}
        for l2 in labels:
            confusion_matrix[l1][l2] = 0

    for tsl, t in tqdm(test):
        algo_kwargs['t'] = t
        pred_label, min_score = -1, math.inf
        if prunning:
            algo_kwargs['prune_score'] = math.inf

        for trl, s in train:
            algo_kwargs['s'] = s
            score = algo_func(**algo_kwargs)
            if score < min_score:
                min_score = score
                pred_label = trl
                if prunning:
                    algo_kwargs['prune_score'] = min_score

        confusion_matrix[tsl][pred_label] = confusion_matrix[tsl][pred_label] + 1

        y_pred.append(tsl == pred_label)

    return y_pred, confusion_matrix

def parse_args(argv):
    shortopts = 'm:d:a:w:ph'

    longopts = [
        'metric=',
        'dimensions=',
        'algorithm=',
        'warp=',
        'prunning',
        'help'
    ]

    config = Config()
    options, _ = getopt.getopt(sys.argv[1:], shortopts, longopts)

    for opt, arg in options:
        if opt in ('-m', '--metric'):
            if arg == 'absolute':
                config.algo_kwargs['dist'] = abs_distance
            elif arg == 'euclidean':
                config.algo_kwargs['dist'] = euclidean_distance
        if opt in ('-d', '--dimensions'):
            if arg in ('1' , '3'):
                config.dimensions = int(arg)
        elif opt in ('-a', '--algorithm'):
            if arg == 'naive':
                config.algo_func = dtw_naive
            elif arg == 'sakoe_chiba':
                config.algo_func = dtw_sakoe_chiba
        elif opt in ('-w', '--warp'):
            config.algo_kwargs['w'] = int(arg)
        elif opt in ('-p', '--prunning'):
            config.prunning = True
        elif opt in ('-h', '--help'):
            config.show_help = True

    return config


def print_help():
    print("""Dynamic Time Warping Algorithm.
Usage:
    python main.py -d 1 -m euclidean -a naive
    python main.py -d 3 -m absolute -a sakoe_chiba -w 7 -p

Options:
    -a --algorithm=naive|sakoe_chiba    DTW algorithm
    -d --dimensions=1|3                 Data points dimensions
    -m --metric=absolute|euclidean      Distance metric
    -p --prune                          Enable 1nn pruning
    -w --warp=NUM                       Warp parameter (sakoe_chiba only)
    -h --help                           Print this message
    """)

def main():
    if len(sys.argv) <= 1:
        print('Missing arguments.')
        sys.exit(1)

    cfg = parse_args(sys.argv[1:])
    train, test, labels = None, None, None

    if cfg.show_help:
        print_help()
        sys.exit(0)

    if not cfg.algo_func:
        print('Missing algorithm argument.')
        sys.exit(1)

    if cfg.dimensions == 1:
        train, test, labels = read_input_1d()
    elif cfg.dimensions == 3:
        train, test, labels = read_input_3d()
    else:
        print('Missing dimensions argument.')
        sys.exit(1)

    if 'dist' not in cfg.algo_kwargs:
        print('Missing metric argument.')
        sys.exit(1)

    if cfg.algo_func == dtw_sakoe_chiba:
        if 'w' not in cfg.algo_kwargs:
            print('Missing warp argument.')
            sys.exit(1)
        elif cfg.algo_kwargs['w'] <= 0:
            print('Invalid warp argument (it must be > 0).')
            sys.exit(1)

    y_pred, confusion_matrix = predict(train, test,
                                       cfg.algo_func,
                                       cfg.algo_kwargs,
                                       cfg.prunning)

    print(sum(y_pred), len(y_pred), sum(y_pred) / len(y_pred) * 100 )

    # ACTUAL \ PREDICTED
    import seaborn as sns
    import matplotlib.pyplot as plt
    order_labels = sorted(map(int, confusion_matrix.keys()))
    labels = [labels[key] for key in order_labels]
    cm = []
    sys.stdout.write('\t')
    for l1 in order_labels:
        sys.stdout.write(str(l1) + '\t')
        cm.append([])
    sys.stdout.write('\n')
    for l1 in order_labels:
        sys.stdout.write(str(l1) + '\t')
        for l2 in order_labels:
            cm[l1-1].append([])
            cm[l1-1][l2-1] = confusion_matrix[l1][l2]
            sys.stdout.write(str(confusion_matrix[l1][l2]) + '\t')
        sys.stdout.write('\n')

    if cfg.algo_func == dtw_sakoe_chiba:
        algo = 'sakoe_chiba'
    else:
        algo = 'naive'

    if cfg.dimensions == 1:
        d = '1'
    else:
        d = '3'

    if cfg.prunning:
        p = '1'
    else:
        p = '0'  
    
    if 'w' in cfg.algo_kwargs:
        w = str(cfg.algo_kwargs['w'])
    else:
        w = '0'

    sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.savefig('{}_d{}_p{}_w{}.png'.format(algo, d, p, w))

if __name__ == '__main__':
    main()
