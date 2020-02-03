# coding: utf-8

import os, io, time, datetime, io, sys
import re
import bs4
import numpy as np
import pandas as pd
import tushare as ts

from _base_reader import (_base_reader)
from _utils import(_mkdir_cache, _read_buffer)
from _network import(_endless_get)
from _const import(DATABASE_FILENAME)

TUSHARE_TOKEN = 'f4673f7862e73483c5e65cd9a036eedd39e72d484194a85dabcf958b'

URL_REPORT_T = 'http://money.finance.sina.com.cn/corp/go.php/vDOWN_%s/displaytype/4/stockid/%s/ctrl/all.phtml'
URL_XDR_T    = 'http://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/%s.phtml'

XDR_TAB_MATCH = '分红'
REPORT_DATE = '报表日期'
UNIT = '单位'
PLANE = '进度'
CARRIED = '实施'
CLOSE = 'close'
TS_CODE = 'ts_code'
TS_DATE = 'trade_date'

BOUNS = '送股(股)'
GIFT = '转增(股)'
DONATION = '派息(税前)(元)'

XDR_COLS = [BOUNS, GIFT, DONATION, '进度']
PRICE_COLS = ['open', 'high', 'close', 'low']

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

        self._pro_tushare = None

        # XXX: self._symbols will be changed in self.update()
        if isinstance(self._symbols, (str, int)):
            self._symbols = [self._symbols]

        self._symbols = self._read_symbols(self._symbols)

    def update(self, cb_progress=None, **kwargs):
        '''
        update cache
        @params:
        cb_progress: {function}
            call back function to display progress, this function should have
            prototype of cb_progress(inter, total) inter is {int}, total is {int},
            inter is the current inter point, total is the total points
        category: {list[str], str, None} 
            category to update, list or str of ['info', 'daily', 'report', 'all'] or None(all)
        '''
        self._category = kwargs.pop('category', CATEGORY)

        if self._category is None or self._category == 'all':
            self._category = CATEGORY
        elif isinstance(self._category, str):
            self._category = [self._category]

        def _progress(i, t):
            if not cb_progress is None and callable(cb_progress):
                cb_progress(i, len(self._symbols))

        if INFO in self._category:
            self._cache_info()
            # XXX: always use updated symbols from info
            self._symbols = self._read_symbols(self._symbols)

        # FIXME: multiprocess pool
        if DAILY in self._category:
            for (i, s) in enumerate(self._symbols):
                _progress(i, len(self._symbols))
                self._cache_daily(s)

            _progress(len(self._symbols), len(self._symbols))

        if REPORT in self._category:
            for (i, s) in enumerate(self._symbols):
                _progress(i, len(self._symbols))
                self._cache_url_report(s)

            _progress(len(self._symbols), len(self._symbols))

    def build(self, file=None, **kwargs):
        '''
        build database file
        @params:
        file: {str}
            file to be built, default is cn_database.h5
        update: boolean
            do full-update first
        xdr: {str} or None
            xdr mode, support: ['fill'] or None. given None will return
            the original daily only 
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

        _daily = self._read_daily(xdr=kwargs.pop('xdr', None))
        _daily.to_hdf(file, key=DAILY, mode='a')

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

    def _wrapper_read_html(self, url, encoding='gb2312', log='err', **kwargs):
        '''
        wrapper for pd.read_html(), and log if exception catched
        @params:
        url: {str}
            url to read
        encoding: {str}
            encoding for request
        log: {str}
            log filename
        kwargs:
            see pd_read_html
        '''
        (text, _) = _endless_get(url, None, encoding)
        try:
            df = pd.read_html(text, **kwargs)
        except ValueError:
            # tempz
            if not os.path.exists('log'):
                os.mkdir('log')
            _log = os.path.join('log', log)
            with open(_log, encoding=self._encoding, mode='w') as f:
                f.write(text)

            return None

        return df

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
            symbols = [str(s) for s in symbols if str(s) in all_symbols]
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

        if self._pro_tushare is None:
            self._pro_tushare = ts.pro_api(TUSHARE_TOKEN)

        def _sym_tushare(s):
            if int(s) > 0 and int(s) < 600000:
                s += '.SZ'
            else:
                s += '.SH'

            return s

        _daily = None
        while _daily is None:
            try:
                _daily = self._pro_tushare.daily(ts_code=_sym_tushare(symbol))
            except:
               time.sleep(60) 

        _daily.drop(axis=1, columns=[TS_CODE], inplace=True)
        _daily[TS_DATE] = _daily[TS_DATE].apply(lambda x: re.sub(r'^(\d\d\d\d)(\d\d)(\d\d)', r'\1-\2-\3', x))
        _daily.set_index(keys=TS_DATE, inplace=True)

        file = os.path.join(self._path[DAILY], symbol, 'daily.csv')
        _daily.to_csv(file, sep=',', encoding=self._encoding)


        # xdr
        xdr = self._wrapper_read_html(URL_XDR_T % symbol, log='xdr_%s.log' % (symbol), \
                                      header=2, index_col=5, \
                                      attrs={'id': 'sharebonus_1'}, parse_dates=True)[0]

        if xdr is None:
            raise OSError('%s: request xdr failed' % symbol)

        xdr = xdr.iloc[slice(None), 1:5]
        xdr.columns = XDR_COLS

        try:
            xdr = xdr[xdr[PLANE] == CARRIED]
        except TypeError:
            # xdr is possibly only one invalid line
            pass

        xdr.sort_index(ascending=False, inplace=True) # descending for easy-reading
        xdr.dropna(how='all', inplace=True)
        file = os.path.join(self._path[DAILY], symbol, 'xdr.csv')

        xdr.iloc[slice(None), 0:3].to_csv(file, sep=',', encoding=self._encoding)

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

    def _read_daily(self, xdr=None):
        '''
        read daily Dataframe
        the output Dataframe:
                                    subjects('open', 'close' ...)
            symbols date(datetime)
            000001  1991-01-01
        @params:
        xdr: {str} or None
            xdr mode, support: ['fill'] or None. given None will return
            the original daily only 
        '''
        tables = []
        for s in self._symbols:
            t = self._read_one_daily(s, xdr=xdr)
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

    def _read_one_daily(self, symbol, xdr=None):
        '''
        read-in one daily, in *ascending*
        the output Dataframe:
                                subjects('open', 'close' ...)
            date(datetime)
            1991-01-01
        @params:
        symbol: {str}
            symbol to read        
        xdr: {str} or None
            xdr mode, support: ['fill'] or None. given None will return
            the original daily only 
        '''
        file = os.path.join(self._path[DAILY], symbol, 'daily.csv')

        if not os.path.exists(file):
            raise OSError('daily file: %s does not exist' % file)

        # FIXME: integrate
        daily = pd.read_csv(file, header=0, index_col=0, 
                            parse_dates=True, encoding=self._encoding)
        daily.sort_index(inplace=True)

        # xdr things
        if xdr is not None:
            if not xdr in ['fill']:
                raise NotImplementedError('%s is not supported' % xdr)

            file = os.path.join(self._path[DAILY], symbol, 'xdr.csv')

            # FIXME: warning instead?
            if not os.path.exists(file):
                raise OSError('xdr file %s does not exist' % file)

            xdr_df = pd.read_csv(file, header=0, index_col=0,
                                 parse_dates=True, encoding=self._encoding)
            xdr_df.sort_index(inplace=True)

            if xdr_df.loc[xdr_df.index.duplicated(keep=False)].empty == False:
                first = xdr_df.loc[xdr_df.index.duplicated(keep='first')]
                last  = xdr_df.loc[xdr_df.index.duplicated(keep='last')]

                dup = first + last

                xdr_df.loc[dup.index] = dup

                xdr_df = xdr_df.loc[~xdr_df.index.duplicated(keep='first')]

            # XXX: workaround for non-date index
            xdr_df.index = pd.to_datetime(xdr_df.index, errors='coerce')
            xdr_df = xdr_df.loc[xdr_df.index.dropna()]

            # concat to append xdr dates
            daily = pd.concat([daily, xdr_df], axis=1)

            for e in list(xdr_df.index.values):

                _bouns = xdr_df.loc[e, BOUNS] / 10
                _gift  = xdr_df.loc[e, GIFT] / 10
                _dona  = xdr_df.loc[e, DONATION] / 10

                if np.isnan(daily.loc[e, CLOSE]):
                    daily.loc[e, PRICE_COLS] = daily.shift(1).loc[e, PRICE_COLS]
                    daily.loc[e, PRICE_COLS] = daily.loc[e, PRICE_COLS].apply(lambda x: (x - _dona) / (1 + _gift + _bouns))
                
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

            def __sanitize(lines):

                # FIXME: workaround for 002886: '20141231.1',
                #        use dot version?
                r = re.compile(r'^(\d+)\.\d+,')

                return [l for l in lines if not r.match(l)]

            all_lines = __sanitize(_read_buffer(file=file, encoding=self._encoding))

            # FIXME: integrated
            t = pd.read_csv(io.StringIO(''.join(all_lines)), \
                            header=0, index_col=0, parse_dates=True, encoding=self._encoding)

            tables.append((t, table_type))

        _report = pd.concat([_[0] for _ in tables], axis=1, keys=[_[1] for _ in tables],
                            names=[TABLE_TYPE, SUBJECTS])
        _report.index.name = DATE

        return _report
