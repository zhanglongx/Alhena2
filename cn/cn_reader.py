# coding: utf-8

import os, time, io, sys
import re
import numpy as np
import pandas as pd
import tushare as ts

from _base_reader import (_base_reader)
from _utils import(_mkdir_cache, _read_buffer)
from _network import(_endless_get)
from _const import(DATABASE_FILENAME)

URL_TEMPLATE = 'http://money.finance.sina.com.cn/corp/go.php/vDOWN_%s/displaytype/4/stockid/%s/ctrl/all.phtml'

CN     = 'cn'
INFO   = 'info'
DAILY  = 'daily'
REPORT = 'report'

CATEGORY = [INFO, DAILY, REPORT]

SYMBOLS    = 'symbols'
TABLE_TYPE = 'table_type'
SUBJECTS   = 'subjects'
DATE       = 'date'

class cn_reader(_base_reader):
    '''
    Parameters
    ----------
    path: {str}
        Alhena2 path
    symbols : {list[(str, int)], int, str, None}
        String symbol of like of symbols. If symbols is int or list[int], at-least has
        Python type 
    start: string, (defaults to '1/1/2007')
        Starting date, timestamp. Parses many different kind of date
        representations (e.g., 'JAN-01-2010', '1/1/10', 'Jan, 1, 1980')
    end: string, (defaults to today)
        Ending date, timestamp. Same format as starting date.
    skip_time: int, (default to 30)
        skip update if cache file modification time is less than skip_time, 
        in days
    skip_size: int, (default to 10)
        skip update if cache file size is less than skip_size, in KB
    '''
    def __init__(self, path, symbols=None, start='1/1/2007', end=None, **kwargs):
        super().__init__(path=path, symbols=symbols, start=start, end=end, **kwargs)

        (_path_cn, _path_info, _path_daily, _path_report) = _mkdir_cache(self._root, CN, CATEGORY)

        self._path[CN]     = _path_cn
        self._path[INFO]   = _path_info
        self._path[DAILY]  = _path_daily
        self._path[REPORT] = _path_report

        # network utils
        self._skip_time = kwargs.pop('skip_time', 30)
        self._skip_size = kwargs.pop('skip_size', 10)

        # XXX: self._symbols will be changed in self.update()
        if isinstance(self._symbols, (str, int)):
            self._symbols = [self._symbols]

        self._symbols = self._read_symbols([str(s) for s in self._symbols])

    def update(self, **kwargs):
        '''
        update cache
        @params:
        category: {list[str], str, None} 
            category to update, list or str of ['info', 'daily', 'report'] or None(all)
        '''
        self._category = kwargs.pop('category', CATEGORY)

        if self._category is None:
            self._category = CATEGORY
        elif isinstance(self._category, str):
            self._category = [self._category]

        if INFO in self._category:
            self._cache_info()
            # XXX: always use updated symbols from info
            self._symbols = self._read_symbols(self._symbols)

        # FIXME: multiprocess pool
        if DAILY in self._category:
            print('cn-daily update has not been implemented', file=sys.stderr)

        if REPORT in self._category:
            for (i, s) in enumerate(self._symbols):
                self._cache_url_report(s)

    def build(self, file=None, **kwargs):
        '''
        build database file
        @params:
        file: {str}
            file to be built, default is cn_database.h5
        update: boolen
            do full-update first
        -------------
        database file has three Dataframe, as below
        Dataframe info:
                      name, industry  
            symbols  
            000001  
        Dataframe report:
                                      table_type 
                                      subject
            symbols  date(datetime)
            000001   1991-01-01
        '''
        if kwargs.pop('update', False):
            self.update(category=None)

        if file is None:
            file = os.path.join(self._path[CN], DATABASE_FILENAME)

        # XXX: auto detect file extension
        _info = self._read_info()
        _info.to_hdf(file, key=INFO, mode='w')

        _report = self._read_reports()
        _report.to_hdf(file, key=REPORT, mode='a')

    def _is_need_udpate(self, file):
        '''
        is a cache file need to be updated
        @params:
        file:
            file to check
        '''
        if not os.path.exists(file):
            return True

        # file mtime
        mtime = os.path.getmtime(file)
        if time.time() - mtime > self._skip_time * (60 * 60 * 24):
            return True

        # file size
        size = os.stat(file).st_size
        if size < self._skip_size * 1024:
            return True

        return False

    def _read_symbols(self, symbols=None):
        '''
        read in symbols
        @params:
        symbols: {List[str], None}
            symbols to be checked, or None to use internal(usually all) 
        '''
        info_file = os.path.join(self._path[INFO], '%s.csv' % INFO)

        # FIXME: always update info? currently may ignore some
        #        very new input symbols
        if not os.path.exists(info_file):
            self._cache_info()

        # read from list
        lines = _read_buffer(info_file, encoding=self._encoding)

        all_symbols = []
        # skip info head
        for l in lines[1:]:
            all_symbols.append(str(l.split(',')[0]))

        if not symbols is None:
            symbols = [str(s) for s in symbols if s in all_symbols]
            assert symbols # prevent ill-input, but can be removed

            return symbols

        return all_symbols

    def _cache_info(self):
        '''
        cache info file from tushare
        '''
        file = os.path.join(self._path[INFO], '%s.csv' % INFO)

        if not self._is_need_udpate(file):
            return

        try:
            ts_id = ts.get_stock_basics()
        except:
            raise ConnectionError('getting info from tushare failed')

        ts_id.sort_index(inplace=True)

        # fix prefix 0
        ts_id.index.astype(str)

        try:
            ts_id.to_csv(file, sep=',', encoding=self._encoding)
        except:
            raise OSError('writing %s error' % file)

    def _cache_url_report(self, symbol):
        '''
        cache three report files of one symbol
        @params:
        symbol: {str}
            symbol        
        '''
        sym = symbol

        _symbol_folder = os.path.join(self._path[REPORT], sym)
        if not os.path.exists(_symbol_folder):
            os.mkdir(_symbol_folder)

        for table_type in ['BalanceSheet', 'ProfitStatement', 'CashFlow']:
            url = URL_TEMPLATE % (table_type, sym)

            file = os.path.join(_symbol_folder, '%s.csv' % table_type)

            if not self._is_need_udpate(file):
                # FIXME: more check, parse is in Quarter
                return

            while(1):
                (text, raw) = _endless_get(url, None, 'gb2312') # tempz
                # try parse the fisrt line
                first_line = text.split('\n', maxsplit=1)[0]

                r = re.compile(r'^报表日期\s+\d{8}\s+')
                if not r.match(first_line):
                    continue

                break

            # XXX: transpose report file to use parse_dates, see self._read_one_report()
            t = pd.read_csv(io.BytesIO(raw), delim_whitespace=True, header=0, \
                            index_col=0, encoding='gb2312').transpose()

            t.drop('19700101', inplace=True)
            t.drop('单位', axis=1, inplace=True)
            t.dropna(axis=1, how='all', inplace=True)

            t.to_csv(file, sep=',', encoding=self._encoding)

    def _read_info(self):
        '''
        read info into Dataframe
        '''
        info_file = os.path.join(self._path[INFO], '%s.csv' % INFO)

        if os.path.exists(info_file):
            t = pd.read_csv(info_file, header=0, dtype={'code': str}, \
                            encoding=self._encoding)
            t.rename({'code': SYMBOLS}, axis=1, inplace=True)
            t.set_index(SYMBOLS, inplace=True)
        else:
            raise OSError('info file: %s not exists' % info_file)

        return t

    def _read_reports(self):
        '''
        read reports and output Dataframe
        the output Dataframe:
                                    table_type
                                    subjects
            symbols  date(datetime)
            000001   1991-01-01
        ''' 
        tables = []
        for s in self._symbols:
            t = self._read_one_report(s)
            tables.append((t, s))

        all_reports = pd.concat([_[0] for _ in tables], keys=[_[1] for _ in tables], \
                                names=[SYMBOLS, DATE]) 

        return all_reports

    # XXX: deprecated for Alhena1 daily database only
    def __inv_ex(self, df_ex, df_daily):
        '''
        Parameters
        ----------
        df_ex: {pd.Dataframe}
            ex info in dataframe format
        df_daily: {pd.Dataframe}
            daily info in dataframe format
        '''

        def __one_inv_ex(series, gift, donation, bouns):
            return series * ( 1 + gift / 10 + donation / 10 ) + bouns / 10

        for xdr_date in df_ex.index.values:

            end = xdr_date - np.timedelta64(1, 'D')

            gift     = (df_ex.loc[xdr_date])['gift']
            donation = (df_ex.loc[xdr_date])['donation']
            bouns    = (df_ex.loc[xdr_date])['bouns']

            df_daily.loc[:end, 'open'] = __one_inv_ex(df_daily.loc[:end, 'open'], \
                                                      gift=gift, donation=donation, bouns=bouns)
            df_daily.loc[:end, 'high'] = __one_inv_ex(df_daily.loc[:end, 'high'], \
                                                      gift=gift, donation=donation, bouns=bouns)
            df_daily.loc[:end, 'low'] = __one_inv_ex(df_daily.loc[:end, 'low'], \
                                                     gift=gift, donation=donation, bouns=bouns)
            df_daily.loc[:end, 'close'] = __one_inv_ex(df_daily.loc[:end, 'close'], \
                                                       gift=gift, donation=donation, bouns=bouns)
            # FIXME: vol

        return df_daily

    def _read_one_daily(self, symbol, subjects=None, ex=None, freq='D'):

        if subjects is None:
            subjects = ['close']

        file = os.path.join(self._path[DAILY], symbol + '.csv')

        # FIXME:
        if not os.path.exists(file):
            print(file + ' does not exist', file=sys.stderr)
            return None

        all = self._helper_read_buffer(file)

        # FIXME: version check

        # copied from Alhena
        # about '20' things: workaround for duplicated items
        r  = re.compile(r'^#\s+(\d+-\d+-\d+,[.0-9]+,[.0-9]+,[.0-9]+)')
        r1 = re.compile(r'20\d\d-\d+-\d+')

        lines_ex = []
        i_daily = 0
        for (i, l) in enumerate(all):
            m = r.match(l)
            if m:
                entry = m.group(1)
                if r1.match(entry):
                    lines_ex.append(entry)
            elif i > 0:
                i_daily = i
                break

        lines_daily = all[i_daily:]

        columns = ['open', 'high', 'low', 'close', 'vol', 'equity']
        df_daily = pd.read_csv(io.StringIO(''.join([s + '\n' for s in lines_daily])), header=None, \
                               names=columns, parse_dates=True, encoding=self._encoding)

        # fill time-date gap in original .csv
        df_daily = df_daily.asfreq('d', method='ffill')

        columns = ['gift', 'donation', 'bouns']
        df_ex = pd.read_csv(io.StringIO(''.join([s + '\n' for s in lines_ex])), header=None, \
                            names=columns, parse_dates=True, encoding=self._encoding)

        if ex is None:
            df_daily = self.__inv_ex(df_ex=df_ex, df_daily=df_daily)
        elif ex == 'backward':
            # XXX: df_daily is backward default
            pass
        else:
            raise ValueError('%s is not supported' % ex)

        # combine with ex-info
        df_daily = pd.concat([df_daily, df_ex], axis=1)

        if freq.lower() == 'd':
            # as 'D', only provide original data, no fill
            return df_daily[subjects]
        else:
            return df_daily.asfreq(freq, method='ffill')[subjects]

    def _read_one_report(self, symbol):
        '''
        read-in three table sheets
        the output Dataframe:
                            table_type
                            subjects
            date(datetime)
            1991-01-01
        @params:
        symbol: {str}
            symbol to read 
        '''
        tables = []
        for table_type in ['BalanceSheet', 'ProfitStatement', 'CashFlow']:
            file = os.path.join(self._path[REPORT], symbol, '%s.csv' % table_type)

            # FIXME: maybe pass?
            if not os.path.exists(file):
                raise OSError('report file: %s not exists' % file)

            t = pd.read_csv(file, header=0, index_col=0, parse_dates=True, encoding=self._encoding)\

            tables.append((t, table_type))

        _report = pd.concat([_[0] for _ in tables], axis=1, keys=[_[1] for _ in tables],
                            names=[TABLE_TYPE, SUBJECTS])
        _report.index.name = DATE

        return _report