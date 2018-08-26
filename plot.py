# coding

import argparse
import matplotlib.pyplot as plt

from Alhena2.cn.cn_extractor import (cn_extractor)

def plot(path, save_csv=True, formula=None, symbols=None, start=None, asfreq='A-DEC'):

    data = cn_extractor('.', symbols=symbols, subjects=formula, add_group='industry').gen_data()
    
    data = data.loc[start:].asfreq(asfreq)

    if save_csv is True:
        data.to_csv('t.csv', encoding='gb2312') 
        return

    # save_csv is False
    plt.figure()

    for s in set(data.columns.get_level_values(0)):
        data[s].plot.bar(title=s)

    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='''wrapper for Alhena2 extractor''')

    parser.add_argument('-c', '--csv', dest='csv', action='store_true', help='csv output')
    parser.add_argument('-f', '--formula', type=str, nargs='?', help='formula or file input (.json)')
    parser.add_argument('-p', '--path', default='.', type=str, nargs='?', help='Alhena2 path')
    parser.add_argument('-s', '--start', default='2013-01-01', type=str, help='start date')
    parser.add_argument('symbols', default=None, type=str, nargs='*', help='symbols to extract')

    args = parser.parse_args()

    csv     = args.csv
    formula = args.formula
    path    = args.path
    start   = args.start
    symbols = args.symbols

    plot(path=path, save_csv=csv, formula=formula, symbols=symbols, start=start)