# coding: utf-8

import os

from abc import (ABCMeta, abstractmethod)

from _const import (DATABASE_DIRNAME)
from _utils import (_sanitize_dates)

class _base_reader():
    """
    Parameters
    ----------
    path: {str}
        Alhena2 path
    symbols : {List[str], None}
        String symbol of like of symbols
    start : string, (defaults to '1/1/2007')
        Starting date, timestamp. Parses many different kind of date
        representations (e.g., 'JAN-01-2010', '1/1/10', 'Jan, 1, 1980')
    end : string, (defaults to today)
        Ending date, timestamp. Same format as starting date.
    """
    __metaclass__ = ABCMeta
    def __init__(self, path, symbols, start=None, end=None, **kwargs):

        self._path = dict({'base': path})
        self._root = self._check_root_path(path)

        # leave to subclasses
        self._symbols = symbols

        (start, end) = _sanitize_dates(start, end)
        self._start = start
        self._end   = end

        self._encoding = 'utf-8'

    @abstractmethod
    def update(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def build(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _check_root_path(path):
        root = os.path.join(path, DATABASE_DIRNAME)

        if not os.path.exists(root):
            raise OSError('%s not exists, mkdir \'%s first' % (root, DATABASE_DIRNAME))

        return root