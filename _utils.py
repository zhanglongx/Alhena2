import datetime as dt

import os
import requests
import numpy as np
from pandas import to_datetime
from numbers import Number
from requests_file import FileAdapter
from requests_ftp import FTPAdapter

from _network import (_get_one_url)

class SymbolWarning(UserWarning):
    pass

class RemoteDataError(IOError):
    pass

def _mkdir_cache(path, prefix, folders):
    path = os.path.join(path, prefix)
    if not os.path.exists(path):
        os.mkdir(path)

    all = [path]
    for f in folders:
        dir = os.path.join(path, f)
        if not os.path.exists(dir):
            os.mkdir(dir)

        all.append(dir)

    return tuple(all)

# below are copyed from pandas_datareader._utils

def _sanitize_dates(start, end):
    """
    Return (datetime_start, datetime_end) tuple
    if start is None - default is 2007/01/01
    if end is None - default is today
    """
    if is_number(start):
        # regard int as year
        start = dt.datetime(start, 1, 1)
    start = to_datetime(start)

    if is_number(end):
        end = dt.datetime(end, 1, 1)
    end = to_datetime(end)

    if start is None:
        start = dt.datetime(2007, 1, 1)
    if end is None:
        end = dt.datetime.today()
    if start > end:
        raise ValueError('start must be an earlier date than end')
    return start, end

def _read_buffer(file, encoding='utf-8'):
    '''
    Read a file into buffer
    @params:
    file: str
        file to be opened
    '''
    if not os.path.exists(file):
        raise OSError('file %s not exists' % file)

    with open(file, mode='r', encoding=encoding) as f:
        lines = f.readlines()

    return lines

def _init_session(session, retry_count=3):
    if session is None:
        session = requests.Session()
        session.mount('file://', FileAdapter())
        session.mount('ftp://', FTPAdapter())
        # do not set requests max_retries here to support arbitrary pause
    return session

def is_number(obj):
    """
    Check if the object is a number.

    Parameters
    ----------
    obj : The object to check.

    Returns
    -------
    is_number : bool
        Whether `obj` is a number or not.

    Examples
    --------
    >>> is_number(1)
    True
    >>> is_number("foo")
    False
    """

    return isinstance(obj, (Number, np.number))
