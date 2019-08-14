# coding: utf-8

import os
import json
import argparse as arg
import numpy as np
import pandas as pd

from sklearn.preprocessing import scale

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, LSTM
from keras.utils import np_utils

from cn.cn_querier import (cn_info, cn_report)

LSTM_X_FILE = 'lstm.x.npy'
LSTM_Y_FILE = 'lstm.y.npy'

AMOUNT = 'amount'
CLOSE = 'close'
LABEL = 'label'
PERCENT = 'percent'

# mainly for debugging
pd.set_option('display.max_columns', None)

class data_set():
    def __init__(self, keys, use_cache=True, category=None, start=None, end=None, **kwargs):
        '''
        @params
        keys: {dict}
            use `keys` to lookup columns
        use_cache: {boolen}
            use cached x, y if exists
        category: {str, None}
            use `category` to lookup symbols
        start: {str}
            start date
        end: {str}
            end date
        win: int
            rolling windows size
        bins: {int}
            bins to split label Y
        '''
        self._win = kwargs.pop('win', 10) 

        self.use_cache = False
        if use_cache and \
           os.path.exists(LSTM_X_FILE) and os.path.exists(LSTM_Y_FILE):

            print('Using cache: %s, %s' % (LSTM_X_FILE, LSTM_Y_FILE))
            self.use_cache = True
            return 

        _path = '.'

        self._keys = keys

        self._symbols = cn_info(path=_path).get(key=category)

        _report  = cn_report(path=_path, symbols=self._symbols, start=start, end=end, \
                             TTM=True, quarter=None, language='CN')
        self._report = _report.get(formulas=self._keys)

        bins = kwargs.pop('bins', 16)
        self._percent = self.__percent(bins=bins)
        self._report.drop(labels=AMOUNT, axis=1, inplace=True)

    def load(self):
        '''
        load data set cache if use_cache on, or gen new
        '''
        if self.use_cache:
            (x, y) = (np.load(LSTM_X_FILE), np.load(LSTM_Y_FILE))
            print('x y shape:')
            print(x.shape)
            print(y.shape)
        else:
            (x, y) = self.__gen(win=self._win)

            np.save(LSTM_X_FILE, x)
            np.save(LSTM_Y_FILE, y)

        self.x = x
        self.y = y

    def processing(self):
        '''
        processing data set
        '''
        self.__normalized_x()
        self.__onehot()

        return (self.x, self.y)

    def __gen(self, win=10):
        '''
        load (_x, _y) as _x: rolling window, _y labeled percent
        @param:
        win: {int}
            rolling window size
        '''
        _report = self._report

        def __symbols(x):
            # rolling window
            arr = np.array([x.shift(s).values[-win::1][:win] for s in range(len(x))[::-1]])
            arr[np.isinf(arr)]   = np.NaN
            arr[np.isneginf(arr)] = np.NaN

            s = x.index[0][0]

            return ([a for a in arr if len(a[~np.isnan(a).any(axis=1)]) == win], s)

        _x = list()
        _y = list()
        for (w, s) in _report.groupby(level=0).apply(__symbols):
            _x.extend(w)
            _y.extend([self._percent[s] for _ in range(len(w))])

        return (np.array(_x), np.array(_y))

    def __percent(self, bins=16):
        '''
        return labeled Y, MAY DROP some original data
        @parmas
        bins: {int}
            bins spitted with `bins`
        '''
        _report = self._report.copy()

        total = pd.DataFrame()

        def __total(x):
            x[AMOUNT].replace(to_replace=0, value=np.NaN, inplace=True)
            x[AMOUNT].fillna(method='bfill', inplace=True)
            x[AMOUNT].fillna(method='ffill', inplace=True)
            return x[AMOUNT].iloc[-1] / x[AMOUNT].iloc[0]

        total[PERCENT] = _report.groupby(level=0).apply(__total)
        total.dropna(how='any', inplace=True)

        _, bins = np.histogram(total, bins=bins)

        def __bins(x):
            for (i, b) in enumerate(bins[:-1]):
                if x[PERCENT] >= b:
                    r = i
                else:
                    break

            return r

        total[LABEL] = total.apply(__bins, axis=1)

        return total[LABEL]

    def __normalized_x(self):
        (batches, _, features) = self.x.shape

        s = scale(self.x.reshape(batches * self._win, features))

        self.x = s.reshape(batches, self._win, features)

    def __onehot(self):
        self.y = np_utils.to_categorical(self.y)

class model():
    def __init__(self, input_shape, num_classes, batches=100, epochs=50, split=0.2, verbose=1):

        self._batches = batches
        self._epochs  = epochs
        self._split   = split
        self._verbose = verbose

        # FIXME: add timesteps check
        _model = Sequential()
        _model.add(Conv1D(100, 2, activation='relu', input_shape=input_shape))
        _model.add(Conv1D(300, 2, activation='relu'))
        _model.add(MaxPooling1D(3))
        _model.add(LSTM(200, return_sequences=True))
        _model.add(LSTM(200, return_sequences=True))
        _model.add(GlobalAveragePooling1D())
        _model.add(Dropout(0.5))
        _model.add(Dense(num_classes, activation='softmax'))

        _model.compile(loss='categorical_crossentropy',
                       optimizer='adam', metrics=['accuracy'])

        if self._verbose:
            print(_model.summary())

        self._model = _model

    def fit(self, x, y):
        # cb_list = [keras.callbacks.ModelCheckpoint(
        #                 filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        #                 monitor='val_loss', save_best_only=True),
        #             keras.callbacks.EarlyStopping(monitor='acc', patience=10)]

        cb_list = [keras.callbacks.ModelCheckpoint(
                        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
                        monitor='val_loss', save_best_only=True)]

        self._model.fit(x, y, \
                        batch_size=self._batches,
                        epochs=self._epochs,
                        callbacks=cb_list,
                        validation_split=self._split,
                        verbose=self._verbose)

def main():
    parser = arg.ArgumentParser(description='''LSTM under Alhena2''')

    parser.add_argument('-c', '--category', type=str, help='category for symbols')
    parser.add_argument('formula', type=str, help='formula for columns')

    _category = parser.parse_args().category
    _formula  = parser.parse_args().formula

    # use default formula
    with open(_formula, encoding='utf-8') as f:
        _formula = json.load(f)

    ds = data_set(keys=_formula, category=_category)
    ds.load()

    (x, y) = ds.processing()

    m = model((x.shape[1], x.shape[2]), y.shape[1])
    m.fit(x, y)

    return ds

if __name__ == '__main__':
    main()