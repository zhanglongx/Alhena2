# -*- coding=utf-8 -*-

import os, re
import pandas as pd

from _base_querier import (_base_report)
from _const import (DATABASE_DIRNAME, DATABASE_FILENAME, COL_CLOSE)

CN     = 'cn'
INFO   = 'info'
DAILY  = 'daily'
REPORT = 'report'

SYMBOLS    = 'symbols'
TABLE_TYPE = 'table_type'
SUBJECTS   = 'subjects'
DATE       = 'date'
PROFIT_TAB = 'ProfitStatement'

class cn_report(_base_report):
    def __init__(self, path, symbols, start=None, end=None, **kwargs):
        '''
        @params
        path: {str}
            Alhena2 path
        symbols : {List[str], None}
            String symbol of like of symbols
        start: string, (defaults to '1/1/2007')
            Starting date, timestamp. Parses many different kind of date
            representations (e.g., 'JAN-01-2010', '1/1/10', 'Jan, 1, 1980')
        end: string, (defaults to today)
            Ending date, timestamp. Same format as starting date.
        TTM: boolen, (default to false)
            TTM mode on, all profit data will be get as TTM
        quarter: {str}, one of ['Mar', 'Jun', 'Sep', 'Dec', None]
            quarter selected
        add_lasted: boolen
            add lasted data (may not be today)
        language: {str}, one of ['CN', 'EN']
            column name language
        '''
        super().__init__(path=path, symbols=symbols, start=start, end=end, **kwargs)

        if isinstance(self._symbols, (str, int)):
            self._symbols = [self._symbols]

        self._database = os.path.join(self._root, CN, DATABASE_FILENAME)
        if not os.path.exists(self._database):
            raise OSError('%s not exists, run `reader.py build` to build one' \
                          % self._database)

        self._TTM_on = kwargs.pop('TTM', False)

        # select quarter
        _quarter = kwargs.pop('quarter', None)
        if _quarter is not None:
            if not _quarter in ['Mar', 'Jun', 'Sep', 'Dec']:
                raise ValueError('%s not supported' % _quarter)

            self._quarter_grp = pd.Grouper(level=-1, freq='A-%s' % _quarter)
        else:
            self._quarter_grp = pd.Grouper(level=-1, freq='Q-Dec')

        self._load_symbols()
        _daily  = self._load_daily()
        _report = self._load_report()

        self._report = pd.concat([_daily, _report], axis=1)

    def get(self, formulas=None, **kwargs):
        '''
        get calculated Dataframe
        @params:
        formulas: {dict, list[str], str}
            column calculate formula
        mode: {str}
            additional mode 
        '''
        return self._formula(formulas=formulas)

    def _load_symbols(self):
        '''
        load info from database. this also will affect self._symbols
        '''
        _info = pd.read_hdf(self._database, INFO)

        # FIXME: warning?
        _symbols = [str(s) for s in self._symbols if str(s) in _info.index.values]
        self._symbols = _symbols

    def _load_daily(self):
        '''
        load daily from database.
        '''
        # XXX: *MUST* be load from reader HDF, with xdr='fill'
        _daily = pd.read_hdf(self._database, DAILY)

        _daily = _daily.groupby([SYMBOLS, self._quarter_grp]).last()

        _daily = _daily.loc[(self._symbols, slice(self._start, self._end)), :]

        return _daily

    def _load_report(self):
        '''
        load report from database, and filled with full-Q
        '''
        _report = pd.read_hdf(self._database, REPORT)

        # XXX: re-sample, fill with full-Q, must run before other operations
        def _resampler(x):
            return x.set_index(DATE).resample('Q').last()

        _report = _report.reset_index(-1).groupby(level=SYMBOLS).apply(_resampler)

        if self._TTM_on:
            _report = self._run_TTM(_report)

        _report = _report.groupby([SYMBOLS, self._quarter_grp]).last()

        _report = _report.loc[(self._symbols, slice(self._start, self._end)), :]

        # FIXME: level 0 is only for TTM now, so drop it for more simple calculating
        _report.columns = _report.columns.droplevel(0)

        return _report

    def _run_TTM(self, report):
        '''
        XXX: side effect: may turn none-NaN into NaN due to shift
        @params:
        report: Dataframe
            Dataframe to operated on
        '''
        _report = report[PROFIT_TAB].copy()

        def __season(x):
            return x - x.shift(1)

        _report = _report.groupby(level=0).apply(__season)

        # A-MAR
        _idx = _report.unstack(0).index
        _idx = _idx[_idx.month == 3]

        _report.loc[(slice(None), _idx), :] = report.loc[(slice(None), _idx), PROFIT_TAB]

        def __add(x):
            return x + x.shift(1) + x.shift(2) + x.shift(3)

        _report = _report.groupby(level=0).apply(__add)
        report[PROFIT_TAB] = _report

        return report

    def _formula(self, formulas=None):
        '''
        calculate a report Dataframe based on formulas
        @params:
        formulas: {dict, list[str], str}
            column calculate formula
        '''
        if isinstance(formulas, (str, list)):
            # nothing to calculate
            return self._report[_formulas]
        elif isinstance(formulas, dict):
            pass
        else:
            raise TypeError('formulas type is not supported')

        _report = self._report

        def __caculate(left, right):

            right = re.sub(r'([^- %+*\/\(\)\d]+)',
                           r'_report["\1"]',
                           right)

            exec('_report["%s"] = %s' % (left, right))

            return _report

        for f in formulas.keys():
            __caculate(f, formulas[f])

        return _report[list(formulas.keys())]