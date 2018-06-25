# coding: utf-8

import os, time, io, sys
import re
import numpy as np
import pandas as pd 
import tushare as ts

from Alhena2._base_reader import (_base_reader)

URL_TEMPLATE = 'http://money.finance.sina.com.cn/corp/go.php/vDOWN_%s/displaytype/4/stockid/%s/ctrl/all.phtml'

class cn_reader(_base_reader):
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
    def __init__(self, path, symbols=None, start='1/1/2007', end=None, retries=-1):
        super().__init__(path=path, symbols=symbols, start=start, end=end, retries=retries)

        (path_cn, path_info, path_daily, path_report) = self._helper_pre_cache('cn', ['info', 'daily', 'report'])

        self.path['cn']     = path_cn
        self.path['info']   = path_info
        self.path['daily']  = path_daily
        self.path['report'] = path_report

        self.skip_time = 30
        self.skip_size = 10

        self.symbols = self._read_symbols(symbols)

    def update(self):
        
        self._cache_info()

        # FIXME: multiprocess pool
        for (i, s) in enumerate(self.symbols):
            self._helper_progress_bar(i, len(self.symbols))
            self._cache_url_report(s)

    def info(self):

        info_file = os.path.join(self.path['info'], 'info.csv')

        if not os.path.exists(info_file):
            raise OSError('info file %s not exist (may update first?)' % file)

        info = pd.read_csv(info_file, header=0, dtype={'code': str}, encoding=self.encoding)
        info.set_index('code', inplace=True)

        return info.loc[self.symbols]

    def daily(self, subjects=None, ex=None, freq='D'):
        
        # tempz
        return self._read_one_daily(self.symbols[0], subjects=subjects, ex=ex, freq=freq)

    def report(self):
        pass

    def _read_symbols(self, symbols=None):

        info_file = os.path.join(self.path['info'], 'info.csv')

        # FIXME: always update info? currently may ignore some 
        #        very new input symbols
        if not os.path.exists(info_file):
            self._cache_info()

        # read from list
        lines = self._helper_read_buffer(info_file)
            
        all_symbols = []
        # skip info head
        for l in lines[1:]:
            all_symbols.append(str(l.split(',')[0]))

        if not symbols is None:
            symbols = [str(s) for s in symbols if s in all_symbols]
            assert len(symbols) # prevent ill-input, but can be removed

            return symbols

        return all_symbols
    
    def _is_need_udpate(self, file):

        if not os.path.exists(file):
            return True

        # mtime
        mtime = os.path.getmtime(file)
        if time.time() - mtime > self.skip_time * (60 * 60 * 24):
            return True

        # size
        size = os.stat(file).st_size
        if size < self.skip_size * 1024:
            return True

        return False

    def _cache_info(self):

        file = os.path.join(self.path['info'], 'info.csv')

        if not self._is_need_udpate(file):
            return

        try:
            id = ts.get_stock_basics()
        except:
            raise ConnectionError('getting info from tushare failed')

        id.sort_index(inplace=True)

        # fix prefix 0
        id.index.astype(str)

        try:
            id.to_csv(file, sep=',', encoding=self.encoding)
        except:
            raise OSError('writing %s error' % file)

    @staticmethod
    def __is_integrity_report(text):
        # try parse the fisrt line
        first_line = text.split('\n', maxsplit=1)[0]

        r = re.compile('^报表日期\s+\d{8}\s+')
        if not r.match(first_line):
            return False

        return True

    def _cache_url_report(self, symbol):

        sym = symbol
        if isinstance(symbol, int):
            sym = str(symbol)

        file = os.path.join(self.path['report'], '%s.csv' % sym)

        if not self._is_need_udpate(file):
            # FIXME: more check, parse is in Quarter
            return

        tables = []
        for type in ['BalanceSheet', 'ProfitStatement', 'CashFlow']:

            url = URL_TEMPLATE % (type, sym)

            while(1):
                (text, raw) = self._helper_get_one_url(url, None, 'gb2312') # tempz
                if self.__is_integrity_report(text):
                    break

            t = pd.read_csv(io.BytesIO(raw), delim_whitespace=True, header=0, \
                            index_col=0, encoding='gb2312').transpose()

            # FIXME: remove N/A second level like '流动' ?
            t.drop('19700101', inplace=True)
            t.drop('单位', axis=1, inplace=True)
            t.dropna(axis=1, how='all', inplace=True)

            tables.append(t)

        pd.concat(tables, axis=1).to_csv(file, sep=',', encoding=self.encoding)

    # XXX: deprecated for Alhena1 daily database only
    def __inv_ex(self, lines_ex, lines_daily):
        '''
        Parameters
        ----------
        lines_ex: {plain text list}
            ex info in text format
        lines_daily: {plain text list}
            daily info in text format
        '''

        columns = ['gift', 'donation', 'bouns']
        df_ex = pd.read_csv(io.StringIO(''.join([s + '\n' for s in lines_ex])), header=None, \
                            names=columns, parse_dates=True, encoding=self.encoding)

        columns = ['open', 'high', 'low', 'close', 'vol', 'equity']
        df_daily = pd.read_csv(io.StringIO(''.join([s + '\n' for s in lines_daily])), header=None, \
                               names=columns, parse_dates=True, encoding=self.encoding)

        def __one_inv_ex(series, gift, donation, bouns):
            return series * ( 1 + gift / 10 + donation / 10 ) + bouns / 10

        for xdr_date in df_ex.index.values:

            end = xdr_date - np.timedelta64(1, 'D')
            select = df_daily.loc[:end]

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

        if subjects == None:
            subjects = ['close']

        file = os.path.join(self.path['daily'], symbol + '.csv')

        # FIXME:
        if not os.path.exists(file):
            sys.stderr.write(file + ' does not exist')
            return None

        all = self._helper_read_buffer(file)

        # FIXME: version check

        # copied from Alhena
        r = re.compile(r'^#\s+(\d+-\d+-\d+,[.0-9]+,[.0-9]+,[.0-9]+)')

        lines_ex = []
        i_daily = 0
        for (i, l) in enumerate(all):
            m = r.match(l)
            if m:
                lines_ex.append(m.group(1))
            elif i > 1:
                i_daily = i
                break 

        lines_daily = all[i_daily:]

        df_daily = self.__inv_ex(lines_ex, lines_daily)

        if freq.lower() == 'd':
            # as 'D', only provide original data, no fill
            return df_daily[subjects]
        else:
            return df_daily.asfreq(freq, method='ffill')[subjects]