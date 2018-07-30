# -*- coding=utf-8 -*-

import os
import pandas as pd

from Alhena2.cn.cn_reader    import (cn_reader)
from Alhena2._base_extractor import (_base_extractor)

ALL_CN = 'all_cn.h5'

class cn_extractor(_base_extractor):
    def __init__(self, path, symbols=None, start='2007-01-01', end=None, subjects=None, add_group=None, asfreq=None):

        super().__init__(path=path, symbols=symbols, start=start, end=end, \
                         add_group=add_group, asfreq=asfreq)

        all_cn = os.path.join(self.path, ALL_CN)

        if not os.path.exists(all_cn):
            raise OSError('%s not exists, run `make build` first' % all_cn)

        self.info    = pd.read_hdf(all_cn, key='info',   mode='r')
        self.daily   = pd.read_hdf(all_cn, key='daily',  mode='r')
        self.reports = pd.read_hdf(all_cn, key='report', mode='r')

        if not self.subjects is None:
            self.reports = self.reports[self.subjects]

        all_symbols = self.info.index

        if self.symbols is None:
            self.symbols = list(all_symbols)
        else:
            for sym in self.symbols:
                if not sym in list(all_symbols):
                    raise KeyError('%s is not in all symbols' % sym)

    def gen_data(self):

        result = self.reports.loc[self.symbols]

        if not self.add_group is None:

            mean = self._group_mean(self.add_group).sort_index()

            objs = [result.loc[s] for s in self.symbols]
            keys = self.symbols

            objs.append(mean)
            keys.append('mean')

            result = pd.concat(objs, keys=keys, names=['symbols', 'date'])

        return result.unstack(level=0)
