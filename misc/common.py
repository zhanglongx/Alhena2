# coding: utf-8

import os
import pandas as pd

from Alhena2.misc.const import file_tickers

def check_path(path):
    if not os.path.exists(os.path.join(path, 'database')):
        raise KeyError('no subdirectory database found in %s' % path)

    return

def input_tickers(path, tickers=[]):
    '''
       return checked tickers, if ticker is empty, return
       a full list, if ticker is none-empty, check if exists
    '''
    list = file_tickers(path)

    if not os.path.exists(list):
        raise OSError(path + 'error')

    if not len(tickers):
        # read from list
        with open(list, mode='r', encoding='utf-8') as f:
            lines = f.readlines()

        for l in lines:
            tickers.append(l.split(',')[0])

        return tickers
    else:
        __all = pd.read_csv(list, delimiter=',', header=None, dtype=str)

        __all_tickers = __all[0].tolist()

        return [s for s in __all_tickers if s in tickers]