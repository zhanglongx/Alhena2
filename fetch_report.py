# coding: utf-8

import os, io, argparse
from __network import *
import pandas as pd

URL_TEMPLATE = 'http://money.finance.sina.com.cn/corp/go.php/vDOWN_%s/displaytype/4/stockid/%d/ctrl/all.phtml'

def checked_stock(path, stock=[]):
    '''
       return checked stocks, if stock is empty, return
       a full list, if stock is none-empty, check if exists
    '''
    list = os.path.join(path, 'misc', 'list.csv')

    if not os.path.exists(list):
        raise OSError(path + 'error')

    if not len(stock):
        # read from list
        with open(list, mode='r', encoding='UTF-8') as f:
            lines = f.readlines()

        for l in lines:
            stock.append(l.split(',')[0])

        return stock
    else:
        __all = pd.read_csv(list, delimiter=',', header=None, dtype=str)

        __all_stock = __all[0].tolist()

        return [s for s in __all_stock if s in stock]

def update_table(path, stock, encoding='utf-8'):

    loc = os.path.join(path, 'database', 'path')

    if not os.path.exsit(loc):
       os.mkdir(loc)

    tables = []
    for type in ['BalanceSheet', 'ProfitStatement', 'CashFlow']:
        url = URL_TEMPLATE % (type, int(stock))

        raw = geturl(url, encoding=encoding)
        
        one_frame = pd.read_csv(io.BytesIO(raw), delim_whitespace=True, header=0, \
                                index_col=0, encoding=encoding).transpose()

        # FIXME: remove N/A second level like '流动' ?
        one_frame.drop('19700101', inplace=True)
        one_frame.drop('单位', axis=1, inplace=True)

        tables.append(one_frame)

        print(one_frame)

def main(path, stock):
    for s in stock:
        update_table(path, s, 'gb2312') # tempz

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''earning fetcher, fetch the newest earning
                                                 ''')
    parser.add_argument('-p', '--path', nargs='?', default='.', help='Alhena2 path')
    parser.add_argument('stock', nargs='*', help='stock list')

    path  = parser.parse_args().path
    stock = checked_stock(path, parser.parse_args().stock)

    main(path, stock)