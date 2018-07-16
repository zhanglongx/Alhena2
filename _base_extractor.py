# -*- coding=utf-8 -*-

import os

from Alhena2._utils import (_sanitize_dates)

class _base_extractor(object):
    def __init__(self, path, symbols=None, start='2007-01-01', end=None, add_group=None, asfreq=None):

        self.path    = path
        self.symbols = symbols

        (start, end) = _sanitize_dates(start, end)
        self.start = start
        self.end   = end

        if isinstance(add_group, bool):
            self.add_group = add_group
        else:
            raise TypeError('add_group is boolen')

        if isinstance(asfreq, str):
            self.asfreq = asfreq
        else:
            raise TypeError('asfreq is str')