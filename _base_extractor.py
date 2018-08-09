# -*- coding=utf-8 -*-

import os
from abc import (ABCMeta, abstractmethod)

from Alhena2._utils import (_sanitize_dates)

class _base_extractor():
    def __init__(self, path, symbols=None, subjects=None, add_group=None):

        self.path     = path
        self.symbols  = symbols
        self.subjects = subjects

        if not (isinstance(add_group, str) or add_group is None):
            raise TypeError('add_group must be str or None')

        self.add_group = add_group

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