# -*- coding=utf-8 -*-

import os
import re
import json
import numpy as np
import pandas as pd

from _base_extractor import (_base_extractor)

ALL_CN = 'all_cn.h5'

class cn_extractor(_base_extractor):
    def __init__(self, path, symbols=None, subjects=None, add_group=None, as_freq='A-DEC'):

        super().__init__(path=path, symbols=symbols, \
                         subjects=subjects, add_group=add_group, as_freq=as_freq)

        all_cn = os.path.join(self.path, ALL_CN)

        if not os.path.exists(all_cn):
            raise OSError('%s not exists, run `make h5` first' % all_cn)

        self.info    = pd.read_hdf(all_cn, key='info',   mode='r')
        self.daily   = pd.read_hdf(all_cn, key='daily',  mode='r')
        self.reports = pd.read_hdf(all_cn, key='report', mode='r')

        self.symbols  = self._symbols()
        self.subjects = self._subjects()

    def gen_data(self):

        if self.as_freq:
            self.reports = self.reports.unstack(level=0).asfreq(self.as_freq).\
                                stack(level=-1).swaplevel(i=0)

        # columns first, may involves group mean
        self.reports = self._formula()

        # TODO: intend for _group_mean(), inf will effect the whole mean
        #       move to add_group? or provide as a option?
        self.reports.replace([np.inf, -np.inf], np.nan, inplace=True)

        result = self.reports.loc[self.symbols]

        # group mean
        if not self.add_group is None:

            if self.add_group == 'exchange':
                self._supplement_info()

            mean = self._group_mean(self.add_group).sort_index()

            objs = [result.loc[s] for s in self.symbols]
            keys = self.symbols

            objs.append(mean)
            keys.append('mean')

            result = pd.concat(objs, keys=keys, names=['symbols', 'date'])

        return result.unstack(level=0)

    def _symbols(self):

        info = self.info

        _symbols = []
        if (self.symbols is None) or (not self.symbols):
            _symbols = list(info.index)
        else:
            for s in self.symbols:
                if s in info.index:
                    _symbols.append(s)
                else:
                    # FIXME: more friendly
                    _symbols.append(info.index[info['name'] == s].tolist()[0])

        if not _symbols:
            raise ValueError

        return _symbols

    def _supplement_info(self):

        info = self.info

        for sym in list(info.index):
            if int(sym) < 300000:
                info.loc[sym, 'exchange'] = '深A'
            elif int(sym) < 600000:
                info.loc[sym, 'exchange'] = '创业板'
            else:
                info.loc[sym, 'exchange'] = '沪A'

        self.info = info

    def _subjects(self):

        _subjects = []
        if (self.subjects is None) or (not self.subjects):
            _subjects = ['PB', 'PE', 'ROE', 'CASH', 'close']

        elif isinstance(self.subjects, str):
            if os.path.exists(self.subjects):
                with open(self.subjects) as f:
                    _subjects = json.load(f)
            else:
                # make list
                _subjects = [self.subjects]

        else:
            _subjects = self.subjects

        return _subjects

    def _formula(self):

        alias = {'格老指数' : 'close * 股本 / (流动资产合计 - 负债合计 - 存货)',
                 '股本数值' : 'close * 股本',
                 '毛利率'   : '(一、营业总收入 - 二、营业总成本) / 一、营业总收入' }

        _reports = self.reports

        def __caculate(reports, left, right):

            right = re.sub(r'([^- %+*\/\(\)\d]+)',
                           r'reports["\1"]',
                           right)

            exec('reports["%s"] = %s' % (left, right))

            return reports

        if isinstance(self.subjects, list):
            _inputs = self.subjects
        elif isinstance(self.subjects, dict):
            _inputs = self.subjects.keys()
        else:
            raise TypeError('subjects type error')

        _subjects = []
        for s in _inputs:
            if s in _reports.columns:
                # instance get
                pass
            elif s in alias.keys():
                _reports = __caculate(_reports, s, alias[s])
            # order is important for 'PEG'
            elif s == 'PEG':
                _reports['PEG'] = _reports['PE'] / (_reports['五、净利润'].groupby(level='symbols').apply(lambda x: x.pct_change()) * 100)
            elif s == 'TTM':
                if not self.as_freq is None:
                    raise KeyError('TTM should be used with as_freq = None')

                _reports['_s1'] = _reports['五、净利润'] - _reports['五、净利润'].shift(1)

                idx = _reports.unstack(0).asfreq('A-MAR').index
                _reports.loc[(slice(None), idx), '_s1'] = _reports.loc[(slice(None), idx), '五、净利润'] 
                _reports['_TTM'] = _reports['_s1'] \
                                 + _reports['_s1'].shift(1) \
                                 + _reports['_s1'].shift(2) \
                                 + _reports['_s1'].shift(3)

                _reports['TTM'] = (_reports['close'] * _reports['股本']) / _reports['_TTM']

            elif isinstance(self.subjects, dict):
                _reports = __caculate(_reports, s, self.subjects[s])
            else:
                raise KeyError('%s is not in any map' % s)

            _subjects.append(s)

        return _reports[_subjects]