# coding: utf-8

import argparse, os
import tushare as ts

def save_tickers(path):

    try:
        id = ts.get_stock_basics()
    except:
        raise ConnectionError('getting basics from tushare failed')

    id.sort_index(inplace=True)

    # fix prefix 0
    id.index.astype(str)

    loc = os.path.join(path, 'database', 'tickers')
    csv = os.path.join(loc, 'tickers.csv')

    if not os.path.exists(loc):
        os.mkdir(loc)

    try:
        id.to_csv(csv, sep=',')
    except:
        raise OSError('writting %s error' % csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''fetch tickers(with all), and other infomations
                                                 ''')
    parser.add_argument('-p', '--path', nargs='?', default='.', help='Alhena2 path')

    path = parser.parse_args().path

    save_tickers(path)