# coding: utf-8

import os, time, datetime, io, sys
import re
import numpy as np
import pandas as pd
import tushare as ts

from _base_reader import (_base_reader)
from _utils import(_mkdir_cache, _read_buffer)
from _network import(_endless_get)
from _const import(DATABASE_FILENAME)

URL_REPORT_T = 'http://money.finance.sina.com.cn/corp/go.php/vDOWN_%s/displaytype/4/stockid/%s/ctrl/all.phtml'
URL_DAILY_T  = 'http://money.finance.sina.com.cn/corp/go.php/vMS_MarketHistory/stockid/%s.phtml?year=%d&jidu=%d'
URL_XDR_T    = 'http://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/%s.phtml'

DAILY_TAB_MATCH = '季度历史交易'
XDR_TAB_MATCH = '分红'
REPORT_DATE = '报表日期'
UNIT = '单位'

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
            category to update, list or str of ['info', 'daily', 'report', 'all'] or None(all)
        '''
        self._category = kwargs.pop('category', CATEGORY)

        if self._category is None or self._category == 'all':
            self._category = CATEGORY
        elif isinstance(self._category, str):
            self._category = [self._category]

        if INFO in self._category:
            self._cache_info()
            # XXX: always use updated symbols from info
            self._symbols = self._read_symbols(self._symbols)

        # FIXME: multiprocess pool
        if DAILY in self._category:
            for (i, s) in enumerate(self._symbols):
                self._cache_daily(s)

        if REPORT in self._category:
            for (i, s) in enumerate(self._symbols):
                self._cache_url_report(s)

    def build(self, file=None, **kwargs):
        '''
        build database file
        @params:
        file: {str}
            file to be built, default is cn_database.h5
        update: boolean
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

    def _is_need_udpate(self, file, skip_time=None, skip_size=None):
        '''
        is a cache file need to be updated
        @params:
        file:
            file to check
        skip_time: {int}
            skip update if cache file modification time is less than skip_time, 
            in days
        skip_size: {int}
            skip update if cache file size is less than skip_size, in KB
        '''
        if not os.path.exists(file):
            return True

        # file mtime
        mtime = os.path.getmtime(file)
        if not skip_time is None and \
           time.time() - mtime > skip_time * (60 * 60 * 24):
            return True

        # file size
        size = os.stat(file).st_size
        if not skip_size is None and size < skip_size * 1024:
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

        if not self._is_need_udpate(file, skip_time=self._skip_time, \
                                    skip_size=self._skip_size):
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

    def _cache_daily(self, symbol, force=False):
        '''
        cache daily data
        @params:
        symbol: {str}
            symbol
        force: {boolean}
            force to update, even if has caches
        '''
        _symbol_folder = os.path.join(self._path[DAILY], symbol)
        if not os.path.exists(_symbol_folder):
            os.mkdir(_symbol_folder)

        old = None
        if force != True:
            try:
                old = self._read_one_daily(symbol)
            except OSError:
                pass

        def __date_range(start, end):
            def __season(m):
                if not m in range(1, 12):
                    raise ValueError('m: %d is not valid' % m)
                return (m-1) // 3 + 1

            e_year   = int(end.split('-')[0])
            s_year   = int(start.split('-')[0])
            s_season = __season(int(start.split('-')[1]))

            for y in reversed(range(s_year, e_year+1)):
                if y == e_year:
                    e_season = __season(int(end.split('-')[1]))
                else:
                    e_season = 4
                for s in reversed(range(1, e_season+1)):
                    if not (y == s_year and s < s_season):
                        yield (y, s)

        today = datetime.datetime.now().strftime('%Y-%m-%d')

        tables = []
        if not old is None:
            start = pd.to_datetime(str(old.index.values[-1])).strftime('%Y-%m-%d')
            tables.append(old)
        else:
            # use XDR to get start date
            # FIXME: Dataframe is messed up
            url = URL_XDR_T % symbol
            (text, _) = _endless_get(url, None, 'gb2312')
            xdr = pd.read_html(text, match=XDR_TAB_MATCH, header=0, 
                               skiprows=1, index_col=0, attrs={'id': 'sharebonus_1'},
                               parse_dates=True)

            _ = pd.to_datetime(str(xdr[0].index.values[-1])).strftime('%Y-%m-%d')
            start = '%s-01-01' % _.split('-')[0]
            
        for (y, s) in __date_range(start=start, end=today):
            url = URL_DAILY_T % (symbol, y, s)

            (text, raw) = _endless_get(url, None, 'gb2312')
            try:
                season = pd.read_html(text, match=DAILY_TAB_MATCH, header=0, 
                                      index_col=0, skiprows=1, parse_dates=True)
            except ValueError:
                # tempz
                if not os.path.exists('log'):
                    os.mkdir('log')
                _log = os.path.join('log', 'daily_%s_%d_%d.log' % (symbol, y, s))
                with open(_log, encoding='gb2312', mode='w') as f:
                    f.write(text)
                continue

            tables.append(season[0])

        file = os.path.join(self._path[DAILY], symbol, 'daily.csv')

        new = pd.concat(tables).drop_duplicates()
        new.sort_index(ascending=False, inplace=True) # descending for easy-reading
        new.to_csv(file, sep=',', encoding=self._encoding)

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
            url = URL_REPORT_T % (table_type, sym)

            file = os.path.join(_symbol_folder, '%s.csv' % table_type)

            if not self._is_need_udpate(file, skip_time=self._skip_time,\
                                        skip_size=self._skip_size):
                # FIXME: more check, parse is in Quarter
                return

            while(1):
                (text, raw) = _endless_get(url, None, 'gb2312') # tempz
                # try parse the fisrt line
                first_line = text.split('\n', maxsplit=1)[0]

                r = re.compile(r'^%s\s+\d{8}\s+' % REPORT_DATE)
                if not r.match(first_line):
                    continue

                break

            # XXX: transpose report file to use parse_dates, see self._read_one_report()
            t = pd.read_csv(io.BytesIO(raw), delim_whitespace=True, header=0, \
                            index_col=0, encoding='gb2312').transpose()

            t.drop('19700101', inplace=True)
            t.drop(UNIT, axis=1, inplace=True)
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

    def _read_daily(self, category='daily'):
        '''
        read daily Dataframe
        the output Dataframe:
                                    subjects('open', 'close' ...)
            symbols date(datetime)
            000001  1991-01-01
        @params:
        category: {str: 'daily', 'xdr' or 'all'}
            category to read, only support 'daily' now
        '''
        tables = []
        for s in self._symbols:
            t = self._read_one_daily(s)
            tables.append((t, s))

        all_daily = pd.concat([_[0] for _ in tables], keys=[_[1] for _ in tables], \
                              names=[SYMBOLS, DATE])

        return all_daily

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

    def _read_one_daily(self, symbol, category='daily'):
        '''
        read-in one daily, in *ascending*
        the output Dataframe:
                                subjects('open', 'close' ...)
            date(datetime)
            1991-01-01
        @params:
        symbol: {str}
            symbol to read        
        category: {str: 'daily', 'xdr' or 'all'}
            category to read, only support 'daily' now
        '''
        if category != 'daily':
            raise NotImplementedError('%s is not supported' % category)

        file = os.path.join(self._path[DAILY], symbol, 'daily.csv')

        if not os.path.exists(file):
            raise OSError('daily file: %s does not exist' % file)

        # FIXME: integrate
        daily = pd.read_csv(file, header=0, index_col=0, 
                            parse_dates=True, encoding=self._encoding)
        daily.sort_index(inplace=True)

        return daily

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

            # FIXME: integrated
            t = pd.read_csv(file, header=0, index_col=0, parse_dates=True, encoding=self._encoding)\

            tables.append((t, table_type))

        _report = pd.concat([_[0] for _ in tables], axis=1, keys=[_[1] for _ in tables],
                            names=[TABLE_TYPE, SUBJECTS])
        _report.index.name = DATE

        return _report