# coding: utf-8

import argparse

from cn.cn_querier import (cn_report)

def main():
    parser = argparse.ArgumentParser(description='''wrapper for Alhena2 querier''')

    parser.add_argument('-c', '--csv', dest='csv', action='store_true', help='csv output')
    parser.add_argument('-f', '--formula', type=str, nargs='?', help='formula or file input (.json)')
    parser.add_argument('-p', '--path', default='.', type=str, nargs='?', help='Alhena2 path')
    parser.add_argument('-s', '--start', default='2013-01-01', type=str, help='start date')
    parser.add_argument('symbols', default=None, type=str, nargs='*', help='symbols to extract')

    _path    = parser.parse_args().path
    _symbols = parser.parse_args().symbols

    report = cn_report(path=_path, symbols=_symbols, start=None, TTM=False, quarter=None)
    df = report.get(formulas={'profit': '五、净利润'})

    df.to_csv('t.csv')

if __name__ == '__main__':
    main()