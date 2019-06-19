# coding: utf-8

import os
import json
import argparse

from cn.cn_querier import (cn_report)

def main():
    parser = argparse.ArgumentParser(description='''wrapper for Alhena2 querier''')

    parser.add_argument('-c', '--csv', dest='csv', action='store_true', help='csv output')
    parser.add_argument('-f', '--formula', type=str, nargs='?', help='formula or file input (.json)')
    parser.add_argument('-p', '--path', default='.', type=str, nargs='?', help='Alhena2 path')
    parser.add_argument('-s', '--start', default='2013-01-01', type=str, help='start date')
    parser.add_argument('--disable-TTM', dest='TTM', default=True, action='store_false', help='TTM off')
    parser.add_argument('symbols', default=None, type=str, nargs='*', help='symbols to extract')

    _formula = parser.parse_args().formula
    _path    = parser.parse_args().path
    _start   = parser.parse_args().start
    _TTM     = parser.parse_args().TTM
    _symbols = parser.parse_args().symbols

    if not _formula is None and os.path.exists(_formula):
        with open(_formula, encoding='utf-8') as f:
            _formula = json.load(f)

    if len(_symbols) == 0:
        _symbols = None

    report = cn_report(path=_path, symbols=_symbols, start=_start, TTM=_TTM, quarter=None, \
                       language='CN')
    df = report.get(formulas=_formula)

    df.to_csv('t.csv')

if __name__ == '__main__':
    main()