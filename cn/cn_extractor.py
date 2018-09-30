# -*- coding=utf-8 -*-

import os
import re
import json
import pandas as pd

from Alhena2._base_extractor import (_base_extractor)

ALL_CN = 'all_cn.h5'

class cn_extractor(_base_extractor):
    def __init__(self, path, symbols=None, subjects=None, add_group=None, as_freq='A-DEC'):

        super().__init__(path=path, symbols=symbols, \
                         subjects=subjects, add_group=add_group, as_freq=as_freq)

        all_cn = os.path.join(self.path, ALL_CN)

        if not os.path.exists(all_cn):
            raise OSError('%s not exists, run `make build` first' % all_cn)

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

        result = self.reports.loc[self.symbols]

        # group mean
        if not self.add_group is None:

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
                 '股本数值' : 'close * 股本'}

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
            elif isinstance(self.subjects, dict):
                _reports = __caculate(_reports, s, self.subjects[s])
            else:
                raise KeyError('%s is not in any map' % s)

            _subjects.append(s)

        return _reports[_subjects]