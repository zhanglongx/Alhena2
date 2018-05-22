# coding: utf-8

import os, io, argparse
import pandas as pd

from Alhena2.misc.common      import (check_path, input_tickers)
from Alhena2.misc.const       import (loc_report)
from Alhena2.misc.network.get import (geturl)

URL_TEMPLATE = 'http://money.finance.sina.com.cn/corp/go.php/vDOWN_%s/displaytype/4/stockid/%s/ctrl/all.phtml'

def save_one(path, ticker, encoding='utf-8'):

    loc = loc_report(path)

    tables = []
    for type in ['BalanceSheet', 'ProfitStatement', 'CashFlow']:
        url = URL_TEMPLATE % (type, str(ticker))

        raw = geturl(url, encoding=encoding)
        
        one_frame = pd.read_csv(io.BytesIO(raw), delim_whitespace=True, header=0, \
                                index_col=0, encoding=encoding).transpose()

        # FIXME: remove N/A second level like '流动' ?
        one_frame.drop('19700101', inplace=True)
        one_frame.drop('单位', axis=1, inplace=True)

        tables.append(one_frame)

        print(one_frame)

def save_tables(path, tickers):
    for t in tickers:
        save_one(path, t, 'gb2312') # tempz

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''earning fetcher, fetch the newest earning
                                                 ''')
    parser.add_argument('-p', '--path', nargs='?', default='.', help='Alhena2 path')
    parser.add_argument('tickers', nargs='*', help='tickers list')

    path = parser.parse_args().path

    check_path(path)

    tickers = input_tickers(path, parser.parse_args().tickers)

    save_tables(path, tickers)