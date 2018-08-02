# coding

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

import Alhena2.cn.cn_extractor as ex

def plot(path, save_csv=True, formula=None, symbols=None, start=None, asfreq='A-DEC'):

    data = ex.cn_extractor('.', symbols=symbols, subjects=formula, add_group='industry').gen_data()
    
    data = data.loc[start:].asfreq(asfreq)

    if save_csv == True:
        data.to_csv('t.csv', encoding='gb2312') 
        return

    plt.figure()

    for s in formula:
        data[s].plot.bar(title=s)

    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='''wrapper for Alhena2 extractor
                                                 ''')
    parser.add_argument('-c', '--csv', default=True, type=bool, help='csv output')
    parser.add_argument('-f', '--formula', type=str, nargs='?', help='formula or file input (.json)')
    parser.add_argument('-p', '--path', default='.', type=str, nargs='?', help='Alhena2 path')
    parser.add_argument('-s', '--start', default='2013-01-01', type=str, help='start date')
    parser.add_argument('symbols', type=str, nargs='+', help='symbols to extract')

    plot('.', formula=['PB', 'ROE1'], symbols=['000651'], start='2014-01-01')
