# -*- coding=utf-8 -*-

import os
import re
import pandas as pd

from Alhena2._base_extractor import (_base_extractor)

ALL_CN = 'all_cn.h5'

class cn_extractor(_base_extractor):
    def __init__(self, path, symbols=None, subjects=None, add_group=None):

        super().__init__(path=path, symbols=symbols, \
                         subjects=subjects, add_group=add_group)

        all_cn = os.path.join(self.path, ALL_CN)

        if not os.path.exists(all_cn):
            raise OSError('%s not exists, run `make build` first' % all_cn)

        self.info    = pd.read_hdf(all_cn, key='info',   mode='r')
        self.daily   = pd.read_hdf(all_cn, key='daily',  mode='r')
        self.reports = pd.read_hdf(all_cn, key='report', mode='r')

        self.symbols  = self._symbols()
        self.subjects = self._subjects()

    def gen_data(self):

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
        if self.symbols is None:
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
        if self.subjects is None:
            _subjects = ['PB', 'PE', 'ROE', 'CASH', 'close']
        else:
            if isinstance(self.subjects, list):
                _subjects = self.subjects
            elif isinstance(self.subjects, str):
                _subjects = [self.subjects]

        return _subjects

    def _formula(self):

        alias = {'ROE1' : 'ROE / PB',
                 }

        _reports = self.reports

        def __caculate(reports, left, right):

            right = re.sub(r'([^- %+*\/\(\)\d]+)',
                           r'reports["\1"]',
                           right)

            exec('reports["%s"] = %s' % (left, right))

            return reports

        _subjects = []
        for s in self.subjects:
            if s in _reports.columns:
                _subjects.append(s)
            elif s in alias.keys():
                _reports = __caculate(_reports, s, alias[s])
                _subjects.append(s)
            else:
                raise NotImplementedError('anonymous formula is not supported')

        return _reports[_subjects]