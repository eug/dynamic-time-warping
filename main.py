import getopt
import math
import sys

from tqdm import tqdm

from dtw import dtw_naive, dtw_sakoe_chiba
from utils.distance import abs_distance, euclidean_distance
from utils.io import read_input_1d, read_input_3d


class Config:
    algo_func = False
    algo_kwargs = {}
    dimensions  = False
    show_help = False

def predict(train, test, algo_func, algo_kwargs):
    y_pred = []
    for tsl, t in tqdm(test):
        algo_kwargs['t'] = t
        min_label, min_score = -1, math.inf
        for trl, s in train:
            algo_kwargs['s'] = s
            score = algo_func(**algo_kwargs)
            if score < min_score:
                min_score = score
                min_label = trl
                algo_kwargs['prune_score'] = min_score
        y_pred.append(tsl == min_label)

    return y_pred

def parse_args(argv):
    shortopts = 'm:d:a:w:h'

    longopts = [
        'metric=',
        'dimensions=',
        'algorithm=',
        'warp=',
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
        elif opt in ('-h', '--help'):
            config.show_help = True
    
    return config


def print_help():
    print("""Dynamic Time Warping Algorithm.
Usage:
    python main.py -d 1 -m euclidean -a naive
    python main.py -d 3 -m absolute -a sakoe_chiba -w 7

Options:
    -a --algorithm=naive|sakoe_chiba    DTW algorithm
    -d --dimensions=1|3                 Data points dimensions
    -m --metric=absolute|euclidean      Distance metric
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
        train, test, labels = read_input_1d()
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

    y_pred = predict(train, test, cfg.algo_func, cfg.algo_kwargs)
    print(sum(y_pred))
    print(len(y_pred))
    print( sum(y_pred) / len(y_pred) * 100 )

if __name__ == '__main__':
    main()
