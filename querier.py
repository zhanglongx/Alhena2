# coding: utf-8

import os
import re
import json
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from cn.cn_querier import (cn_info, cn_report)

def _sanitate_key(key):
    if not isinstance(key, list):
        raise KeyError

    if len(key) == 1:
        return key[0]

    # first as symbols
    _s = [k for k in key if re.match(r'\d+$', k)]

    if len(_s) > 0:
        return _s
    else:
        # TODO
        raise ValueError('only symbols as list supported now')

def main():
    parser = argparse.ArgumentParser(description='''wrapper for Alhena2 querier''')

    parser.add_argument('-c', '--csv', dest='csv', action='store_true', help='csv output')
    parser.add_argument('-d', '--drop', dest='drop', action='store_true', help='drop symbol if only one')
    parser.add_argument('-e', '--en', action='store_true', default=False, help='turn language en on')
    parser.add_argument('-f', '--formula', type=str, nargs='?', help='formula or file input (.json)')
    parser.add_argument('-p', '--path', default='.', type=str, nargs='?', help='Alhena2 path')
    parser.add_argument('-q', '--quarter', default=None, type=str, nargs='?', \
                        help="quarter in ['Mar', 'Jun', 'Sep', 'Dec']")
    parser.add_argument('-s', '--start', default='2000-01-01', type=str, help='start date')
    parser.add_argument('--disable-TTM', dest='TTM', default=True, action='store_false', help='TTM off')
    parser.add_argument('key', default=None, type=str, nargs='+', help='key to extract')

    # runtime
    pd.set_option('display.max_columns', None)

    _drop    = parser.parse_args().drop
    _en      = parser.parse_args().en
    _formula = parser.parse_args().formula
    _path    = parser.parse_args().path
    _quarter = parser.parse_args().quarter
    _start   = parser.parse_args().start
    _TTM     = parser.parse_args().TTM
    _key     = parser.parse_args().key

    if not _formula is None and os.path.exists(_formula):
        with open(_formula, encoding='utf-8') as f:
            _formula = json.load(f)

    language = 'CN' if not _en else 'EN'

    _key =_sanitate_key(_key) 

    if isinstance(_key, str):
        _symbols = cn_info(path=_path).get(key=_key)
    else:
        _symbols = _key

    report = cn_report(path=_path, symbols=_symbols, start=_start, TTM=_TTM, quarter=_quarter, \
                       language=language)
    df = report.get(formulas=_formula)

    def __PEG(df):
        try: 
            df.loc[df['profit%'] < 0, 'profit%'] = np.NaN
            df['PEG'] = df['PE'] / (df['profit%'] * 100)
        except:
            pass

    __PEG(df)

    if len(df.index.levels[0]) == 1 and _drop:
        df.index = df.index.droplevel(0)

    df.to_csv('t.csv')

if __name__ == '__main__':
    main()