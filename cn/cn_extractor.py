# -*- coding=utf-8 -*-

import os
import pandas as pd

from Alhena2.cn.cn_reader    import (cn_reader)
from Alhena2.cn.cn_extractor import (_base_extractor)

ALL_CN = 'all_cn.h5'

class cn_extractor(_base_extractor):
    def __init__(self, path, symbols=None, start='2007-01-01', end=None, add_group=None, asfreq=None):

        super().__init__(path=path, symbols=symbols, start=start, end=end, \
                         add_group=add_group, asfreq=asfreq)

        all_cn = os.path.join(self.path, ALL_CN)

        if os.path.exists(all_cn):
            raise OSError('%s not exists, run `make build` first' % all_cn)

        self.info    = pd.read_hdf(all_cn, key='info',   mode='r')
        self.daily   = pd.read_hdf(all_cn, key='daily',  mode='r')
        self.reports = pd.read_hdf(all_cn, key='report', mode='r')