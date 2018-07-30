# -*- coding=utf-8 -*-

import os
from abc import (ABCMeta, abstractmethod)

from Alhena2._utils import (_sanitize_dates)

class _base_extractor():
    def __init__(self, path, symbols=None, start='2007-01-01', end=None, subjects=None, add_group=None, asfreq=None):

        self.path    = path
        self.symbols = symbols

        (start, end) = _sanitize_dates(start, end)
        self.start = start
        self.end   = end

        self.subjects = subjects

        if not add_group is None:
            if isinstance(add_group, str):
                self.add_group = add_group
            else:
                raise TypeError('add_group is str')

        if not asfreq is None: 
            if isinstance(asfreq, str):
                self.asfreq = asfreq
            else:
                raise TypeError('asfreq is str')

        # implemented in child
        self.info    = None
        self.daily   = None
        self.reports = None

    def _group_mean(self, info_subjects):
        
        self.info.index.name = 'symbols'
        all = self.reports.join(self.info[info_subjects])

        sub = self.info.loc[self.symbols[0], info_subjects]
        for sym in self.symbols:
            if self.info.loc[sym, info_subjects] != sub:
                raise ValueError('symbol catalog is not all same')

        return all[all[info_subjects] == sub].mean(level='date')

    @abstractmethod
    def gen_data(self):
        raise NotImplementedError