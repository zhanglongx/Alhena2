# coding: utf-8

import os
import re
import json
import argparse
import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from cn.cn_querier import (cn_info, cn_report)

def _sanitated_key(path, keys):
    '''
    sanitates keys
    @params:
    path: {str}
        path
    keys: {list}
        keys
    '''
    if not isinstance(keys, list):
        raise TypeError

    _info = cn_info(path=path)

    _symbols = []
    [_symbols.extend(_info.get(key=k)) for k in keys]
    if len(_symbols) >= len(_info.get(key=None)):
        raise ValueError('too many symbols, check input keys')

    return _symbols

class plot():
    def __init__(self, file):
        self._file = file

        with open(self._file, encoding='utf-8') as f:
            self._json = json.load(f)

    def formula(self):
        '''
        get formulas
        '''
        _formula = {}
        for _ in self._json['items']:
            if not isinstance(_['name'], list):
                _formula[_['name']] = _['formula']
            else:
                for i, k in enumerate(_['name']):
                    _formula[k] = _['formula'][i]

        self._formula = _formula

        return self._formula

    def plt(self, df, layout=3):
        '''
        plot items
        @params:
        df: pd.Dataframe
            df contains data to plot
        '''
        _, axes = plt.subplots(nrows=layout, ncols=layout)
        for i, _ in enumerate(self._json['items']):
            if 'kind' not in _:
                continue

            if i >= layout ** 2:
                raise ValueError('items > %d' % layout)

            _ax = axes[int(i//layout), int(i%layout)]

            # kwargs for pd.Dataframe.plot
            kwargs = dict({'kind': _['kind']})
            self.__plot(df=df[_['name']], axes=_ax, **kwargs)

        plt.show()

    @staticmethod
    def __plot(df, axes, **kwargs):
        '''
        wrapper for pd.Dataframe.plot()
        @params:
        df: pd.Dataframe
            df contains data to plot
        axes: plt.axes
            axes to plot
        '''
        title = df.name if isinstance(df, pd.Series) else None
        ax = df.plot(ax=axes, title=title, **kwargs)

        if kwargs.pop('kind', 'line') == 'bar':
            # Make most of the ticklabels empty so the labels don't get too crowded
            ticklabels = ['']*len(df.index)
            # Every 4th ticklabel shows the month and day
            ticklabels[::4] = [item.strftime('%b %d') for item in df.index[::4]]
            # Every 12th ticklabel includes the year
            ticklabels[::12] = [item.strftime('%b %d\n%Y') for item in df.index[::12]]
            ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
            plt.gcf().autofmt_xdate()

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
    parser.add_argument('--plot', action='store_true', default=False, help='plot')
    parser.add_argument('--season_mode', dest='season', default='TTM', help='season mode')
    parser.add_argument('key', default=None, type=str, nargs='*', help='key to extract')

    # runtime
    pd.set_option('display.max_columns', None)

    _drop    = parser.parse_args().drop
    _en      = parser.parse_args().en
    _formula = parser.parse_args().formula
    _path    = parser.parse_args().path
    _quarter = parser.parse_args().quarter
    _start   = parser.parse_args().start
    _toplot  = parser.parse_args().plot
    _season  = parser.parse_args().season
    _key     = parser.parse_args().key

    _symbols = _sanitated_key(path=_path, keys=_key) 
    _plot    = None

    if not _formula is None and os.path.exists(_formula):
        _plot = plot(_formula)
        _formula = _plot.formula()
        _drop = True

    language = 'CN' if not _en else 'EN'

    report = cn_report(path=_path, symbols=_symbols, start=_start, season_mode=_season, quarter=_quarter, \
                       language=language)
    df = report.get(formulas=_formula)

    def __PEG(df):
        try: 
            df.loc[df['PE'] < 0, 'PE'] = np.inf
            df['PEG'] = df['PE'] / (df['profit%'] * 100)
        except:
            pass

    __PEG(df)

    # only one level
    if len(df.index.levels[0]) == 1 and _drop:
        df.index = df.index.droplevel(0)

    if len(_symbols) > 0:
        df = df.groupby(level=-1).median()

    if _toplot and _plot is not None:
        _plot.plt(df=df)

    df.to_csv('t.csv')

if __name__ == '__main__':
    main()