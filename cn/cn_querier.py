# -*- coding=utf-8 -*-

import os, re
import json
import pandas as pd

from _base_querier import (_base_querier)
from _utils import (_sanitize_dates)
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

NAME     = 'name'
INDUSTRY = 'industry'
EXCHANGE = 'exchange'

LANGUAGE_FILE = '../extra/querier_en.json'

class cn_info(_base_querier):
    def __init__(self, path, **kwargs):
        super().__init__(path=path, **kwargs)

        self._database = os.path.join(self._root, CN, DATABASE_FILENAME)
        self._info = self._load_info()

    def get(self, **kwargs):
        '''
        get symbols based on key
        @params
        key: {str}
            query 'symbols', 'name', 'industry', 'exchange' on key
        '''
        _key = kwargs.pop('key', None)

        _idx = self._info.index

        if not _key is None:
            _reset_idx_info = self._info.reset_index()

            if isinstance(_key, int):
                _key = str(_key)
            elif isinstance(_key, str):
                pass
            else:
                raise TypeError('_key type is not supported')

            for k in [SYMBOLS, NAME, INDUSTRY, EXCHANGE]:
                if _key in list(_reset_idx_info[k]):
                    _idx = _reset_idx_info[_reset_idx_info[k] == _key][SYMBOLS]

        return list(_idx)

    def _load_info(self):
        '''
        load info from database
        '''
        _info = pd.read_hdf(self._database, INFO)

        _i_idx = _info.index.astype(int)

        # supplement for exchange
        _info[EXCHANGE] = 'SZ.A'
        _info.loc[_i_idx < 300000, EXCHANGE] = 'SZ'
        _info.loc[600000 <= _i_idx, EXCHANGE] = 'SH'
        _info.loc[688000 <= _i_idx, EXCHANGE] = 'SH.A'

        return _info

class cn_report(_base_querier):
    def __init__(self, path, **kwargs):
        '''
        @params
        path: {str}
            Alhena2 path
        symbols : {List[str], None}
            String symbol of like of symbols
        start: string, (defaults to None)
            Starting date, timestamp. Parses many different kind of date
            representations (e.g., 'JAN-01-2010', '1/1/10', 'Jan, 1, 1980')
        end: string, (defaults to today)
            Ending date, timestamp. Same format as starting date.
        season_mode: {str, None}, if is not None, must be one of ['TTM', 'season']
            season: treat as one season
            ttm: accumulated season as one year
            None: none
        quarter: {str}, one of ['Mar', 'Jun', 'Sep', 'Dec', None]
            quarter selected
        add_lasted: boolen
            add lasted data (may not be today)
        language: {str}, one of ['CN', 'EN']
            column name language
        '''
        super().__init__(path=path, **kwargs)

        self._symbols = kwargs.pop('symbols', None)

        (start, end) = _sanitize_dates(kwargs.pop('start', None), kwargs.pop('end', None))
        self._start = start
        self._end   = end

        if isinstance(self._symbols, (str, int)):
            self._symbols = [self._symbols]

        self._database = os.path.join(self._root, CN, DATABASE_FILENAME)
        if not os.path.exists(self._database):
            raise OSError('%s not exists, run `reader.py build` to build one' \
                          % self._database)

        self._season_mode = kwargs.pop('season_mode', 'TTM')
        if self._season_mode is None or self._season_mode in ['TTM', 'season']:
            pass
        else:
            raise KeyError('%s is not support' % self._season_mode)

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

        if kwargs.pop('language', 'CN') == 'EN':
            _language = os.path.join(self._root, LANGUAGE_FILE)
            if not os.path.exists(_language):
                raise OSError('%s not found' % _language)

            with open(_language) as f:
                _lang_dict = json.load(f)

            self._report = self._formula(formulas=_lang_dict)
        else:
            # warning?
            pass

    def get(self, **kwargs):
        '''
        get calculated Dataframe
        @params:
        formulas: {dict, list[str], str, None}
            column calculate formula, None for whole and no calculating
        mode: {str}
            additional mode 
        '''
        formulas = kwargs.pop('formulas', None)

        return self._formula(formulas=formulas).copy()

    def _load_symbols(self):
        '''
        load info from database. this also will affect self._symbols
        '''
        _info = pd.read_hdf(self._database, INFO)

        if self._symbols is None:
            self._symbols = list(_info.index)
            return

        # FIXME: warning?
        _symbols = [str(s) for s in self._symbols if str(s) in _info.index.values]
        self._symbols = _symbols

    def _load_daily(self):
        '''
        load daily from database.
        '''
        # XXX: *MUST* be load from reader HDF, with xdr='fill'
        _daily = pd.read_hdf(self._database, DAILY).loc[(self._symbols, slice(None))]

        _daily = _daily.groupby([SYMBOLS, self._quarter_grp]).last()

        _daily = _daily.loc[(slice(None), slice(self._start, self._end)), :]

        return _daily

    def _load_report(self):
        '''
        load report from database, and filled with full-Q
        '''
        _report = pd.read_hdf(self._database, REPORT).loc[(self._symbols, slice(None))]

        # XXX: re-sample, fill with full-Q, *MUST* run before other operations
        def _resampler(x):
            return x.set_index(DATE).resample('Q').last()

        _report = _report.reset_index(-1).groupby(level=SYMBOLS).apply(_resampler)

        if self._season_mode is not None:
            _report = self._run_TTM(_report)

        _report = _report.groupby([SYMBOLS, self._quarter_grp]).last()

        _report = _report.loc[(slice(None), slice(self._start, self._end)), :]

        # FIXME: level 0 is only for TTM now, so drop it for more simple calculating
        _report.columns = _report.columns.droplevel(0)

        # workaround for ()
        def __remove_brackets(s):
            return s.replace('(', '（').replace(')', '）')

        _report.rename(mapper=__remove_brackets, axis=1, inplace=True)

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

        if self._season_mode == 'TTM':
            _report = _report.groupby(level=0).apply(__add)
        else:
            # season
            pass

        report[PROFIT_TAB] = _report

        return report

    def _formula(self, formulas=None):
        '''
        calculate a report Dataframe based on formulas
        @params:
        formulas: {dict, list[str], str, None}
            column calculate formula, None for whole and no calculating
        '''
        if formulas is None:
            return self._report
        elif isinstance(formulas, (str, list)):
            # nothing to calculate
            return self._report[formulas]
        elif isinstance(formulas, dict):
            pass
        else:
            raise TypeError('formulas type is not supported')

        _report = self._report

        def __calculate(left, right):

            right = re.sub(r'([^- %+*\/\(\)\d]+)',
                           r'_report["\1"]',
                           right)

            exec('_report["%s"] = %s' % (left, right))

            return _report

        for f in formulas.keys():
            _ = re.match(r'(.*)%', f)
            if _:
                f = _[1]
                __calculate(f, formulas[_[0]])
                _report[_[0]] = _report[f].groupby(level=0).apply(lambda x: x.pct_change())
            else:
                __calculate(f, formulas[f])

        return _report[list(formulas.keys())]