# coding: utf-8

import os

from abc import (ABCMeta, abstractmethod)

from _const import (DATABASE_DIRNAME)

class _base_querier():
    '''
    abstract class for querier
    ----------
    path: {str}
        Alhena2 path
    '''
    __metaclass__ = ABCMeta
    def __init__(self, path, **kwargs):
        '''
        __init__ method, checking path
        @params:
        path: {str}
            Alhena2 path
        '''
        self._root = self._check_root_path(path)

        # FIXME: fixed
        self._encoding = 'utf-8'

    @abstractmethod
    def get(self, **kwargs):
        '''
        abstract method for getting Dataframe
        @params:
        '''
        raise NotImplementedError

    @staticmethod
    def _check_root_path(path):
        root = os.path.join(path, DATABASE_DIRNAME)

        if not os.path.exists(root):
            raise OSError('%s not exists, mkdir \'%s first' % (root, DATABASE_DIRNAME))

        return root