# coding: utf-8

import os
import io

from abc import (ABCMeta, abstractmethod)

from Alhena2._network import (_get_one_url)
from Alhena2._utils   import (_sanitize_dates)

class _base_reader(object):
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
    retries: {int}
        timeout retries, -1 stands forever
    """
    __metaclass__ = ABCMeta
    def __init__(self, path, symbols, start=None, end=None, retries=-1):

        self.path = dict({'base': path})
        self._root = self._check_root_path(path)

        # leave to subclasses
        self.symbols = symbols

        (start, end) = _sanitize_dates(start, end)
        self.start = start
        self.end   = end
        
        if retries < -1:
            retries = -1

        self.retries  = retries

        self.encoding = 'utf-8'

    @abstractmethod
    def update(self):
        raise NotImplementedError

    @abstractmethod
    def info(self):
        raise NotImplementedError

    @abstractmethod
    def daily(self):
        raise NotImplementedError

    @abstractmethod
    def report(self):
        raise NotImplementedError

    def __cb_checked_read_buffer(self):
        return False

    def _checked_read_buffer(self, file):
        '''
            file:
        '''
        if not os.path.exists(file):
            raise OSError('file %s not exists' % file)

        with open(file, mode='r', encoding=self.encoding) as f:
            lines = f.readlines()

        if self.__cb_checked_read_buffer():
            raise OSError('file %s may be corrupted' % file)

        return lines

    def _save_buffer(self, buffer, file):
        # FIXME:
        raise NotImplementedError

    def _helper_pre_cache(self, prefix, folders):
        _path = os.path.join(self._root, prefix)
        if not os.path.exists(_path):
            os.mkdir(_path)

        all = [_path]
        for f in folders:
            dir = os.path.join(_path, f)
            if not os.path.exists(dir):
                os.mkdir(dir)

            all.append(dir)

        return tuple(all)

    def _helper_get_one_url(self, url, param=None, encoding='utf-8'):
        '''
            url:
            param: 
            encoding: {str}
                return result in 'encoding'
        '''
        if not param is None:
            raise NotImplementedError

        while(1):
            (text, raw) = _get_one_url(url, retries=self.retries, \
                                       encoding=encoding)

            break

        return (text, raw)

    @staticmethod
    def _check_root_path(path):
        root = os.path.join(path, 'database')

        if not os.path.exists(root):
            raise OSError('%s not exists' % root)

        return root